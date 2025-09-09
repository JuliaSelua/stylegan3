# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d

import dlib
import cv2
import torch
import numpy as np

import lpips

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def dlib_get_face_embedding(img_tensor):
    """
    img_tensor: torch.Tensor [B, C, H, W], Values in [0,1] or [-1,1]
    return: torch.Tensor [B, 128] Embeddings
    """
    batch_size = img_tensor.shape[0]
    embeddings = []

    # scale values
    if img_tensor.min() < 0:
        img_tensor = (img_tensor + 1) / 2

    for i in range(batch_size):
        # Tensor -> NumPy (uint8, RGB)
        img = img_tensor[i].permute(1, 2, 0).detach().cpu().numpy()
        img_np = np.clip(img * 255, 0, 255).astype(np.uint8)

        # face recogn.
        dets = detector(img_np, 1)
        if len(dets) == 0:
            embeddings.append(torch.zeros(128, dtype=torch.float32, device=img_tensor.device))
            continue

        shape = sp(img_np, dets[0])

        try:
            face_descriptor = facerec.compute_face_descriptor(img_np, shape, num_jitters=0)
        except Exception as e:
            print(f"[WARN] compute_face_descriptor failed on image {i}: {e}")
            embeddings.append(torch.zeros(128, dtype=torch.float32, device=img_tensor.device))
            continue

        # convert to tensor
        face_descriptor = np.array(face_descriptor, dtype=np.float32)
        embeddings.append(torch.tensor(face_descriptor, dtype=torch.float32, device=img_tensor.device))

    return torch.stack(embeddings)


#----------------------------------------------------------------------------
lpips_alex = lpips.LPIPS(net='alex').eval().to(torch.cuda.current_device())

#def style_divergence_loss(img1, img2):
 #   """
 #   LPIPS-based Style-Loss btw img1 and img2.
#    img1, img2: Tensors [B,3,H,W] in [-1,1]
 #   """
 #   device = torch.cuda.current_device()
  #  img1 = img1.to(device)
   # img2 = img2.to(device)
    #return lpips_alex(img1, img2).mean()
#Multi GPUs
class StyleLossHelper:
    def __init__(self, device):
        self.lpips_alex = lpips.LPIPS(net='alex').eval().to(device)

    def __call__(self, img1, img2):
        device = next(self.lpips_alex.parameters()).device
        img1 = img1.to(device)
        img2 = img2.to(device)
        return self.lpips_alex(img1, img2).mean()

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_z2, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0, use_id_loss=True, use_style_loss = True):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.use_id_loss        = use_id_loss
        self.use_style_loss     = use_style_loss
        self.style_loss_fn      = StyleLossHelper(device)


    def run_Gold(self, z, z2, c, update_emas=False):
        ws = self.G.mapping_id(z, c, update_emas=update_emas)
        ws2 = self.G.mapping_style(z2, c, update_emas=update_emas) if z2 is not None else ws.clone()
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping_id(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
                ws2[:, cutoff:] = self.G.mapping_style(torch.randn_like(z2), c, update_emas=False)[:, cutoff:]
                ws = (ws + ws2) / 2
        else:
            #ws = (ws + ws2) / 2
            ws = torch.cat([ws_id[:, :ws_id.shape[1]//2, :], ws_style[:, ws_style.shape[1]//2:, :]], dim=1)

        img = self.G.synthesis(ws, update_emas=update_emas)
        return img, ws
    def run_G(self, z, z2, c, update_emas=False):
        img = self.G(z_id=z, z_style=z2, c=c, update_emas=update_emas)
    
        with torch.no_grad():
            ws_id = self.G.mapping_id(z, c=c, update_emas=update_emas)
            ws_style = self.G.mapping_style(z2, c=c, update_emas=update_emas)
            ws_combined = torch.cat([ws_id, ws_style], dim=2)
    
        return img, ws_combined


    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_z2, gen_c, gain, cur_nimg):
        lambda_id = 0.1
        lambda_style = 0.1
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # Gmain: Maximize logits for generated images + identity loss (as mse)
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_z2, gen_c)             #Generator Forward for whole batch
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)           #GAN loss 
                training_stats.report('Loss/signs/fake', gen_logits.sign())

                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))

                # ID loss on pairs
                if getattr(self, 'use_id_loss', True):
                    # gen_img: [batch_size, C, H, W], 2 img /id
                    emb = dlib_get_face_embedding(gen_img)  # e.g(batch_size, embedding_dim)
                    emb_a = emb[0::2]  # even index: first image of pair
                    emb_b = emb[1::2]  # odd idx: second image of pair
                    # Cosine similarity: 1 - cos(emb_a, emb_b)
                    id_loss = (1 - (emb_a * emb_b).sum(dim=1)).mean()

                    training_stats.report('Loss/G/id_loss', id_loss)
                    loss_Gmain = loss_Gmain + lambda_id * id_loss
                
                if getattr(self, 'use_style_loss', True):
                    img_a = gen_img[0::2] 
                    img_b = gen_img[1::2]  
                    #style_loss = style_divergence_loss(img_a, img_b)  # LPIPS Loss
                    style_loss = self.style_loss_fn(img_a, img_b)
                    training_stats.report('Loss/G/style_loss', style_loss)
                    loss_Gmain = loss_Gmain + lambda_style * style_loss

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if phase in ['Greg', 'Gboth']:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_z2[:batch_size], gen_c[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                loss_Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_z2, gen_c, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
