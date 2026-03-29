"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    SCENE SYNTHESIZER v8.0 (The Environment)                                  ║
║    Objective: Transposing Document into the Physical World                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CORE PRINCIPLES:                                                            ║
║  1. PERSPECTIVE PROJECTION: Realistic 'table-top' viewing angle.             ║
║  2. CONTACT SHADOWS: Soft ambient occlusion at the paper-desk interface.     ║
║  3. MOBILE ISP MODEL: Simulating CMOS sensor noise and lens aberrations.     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
from pathlib import Path

class SceneSynthesizer:
    def __init__(self, desk_path: str):
        self.desk_img = cv2.imread(desk_path)
        if self.desk_img is None:
            # Fallback if image generation fails/path mismatch
            self.desk_img = np.ones((2160, 3840, 3), dtype=np.uint8) * 45 
        
    def add_contact_shadow(self, desk, warped_mask, intensity=0.6):
        """Creates a soft shadow under the paper."""
        # Dilate the mask to create a shadow boundary
        kernel = np.ones((25, 25), np.uint8)
        shadow_mask = cv2.dilate(warped_mask, kernel, iterations=3)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (151, 151), 60)
        
        # Apply darkening
        shadow_factor = (shadow_mask.astype(float) / 255.0) * intensity
        desk_float = desk.astype(float)
        for c in range(3):
            desk_float[:, :, c] *= (1.0 - shadow_factor * 0.4) # Subtle base shadow
            
        return np.clip(desk_float, 0, 255).astype(np.uint8)

    def apply_mobile_distortions(self, img):
        """Simulates lens barrel distortion and vignetting."""
        h, w = img.shape[:2]
        
        # 1. Lens Distortion (Barrel)
        distCoeff = np.zeros((4,1),np.float64)
        distCoeff[0,0] = -0.08 # Barrel distortion
        distCoeff[1,0] = 0.02
        
        cam_matrix = np.eye(3,dtype=np.float32)
        cam_matrix[0,0] = w
        cam_matrix[1,1] = h
        cam_matrix[0,2] = w/2
        cam_matrix[1,2] = h/2
        
        new_cam_matrix, _ = cv2.getOptimalNewCameraMatrix(cam_matrix, distCoeff, (w,h), 1, (w,h))
        mapx, mapy = cv2.initUndistortRectifyMap(cam_matrix, distCoeff, None, new_cam_matrix, (w,h), 5)
        img = cv2.remap(img, mapx, mapy, cv2.INTER_LANCZOS4)
        
        # 2. Vignetting
        X_resultant_kernel = cv2.getGaussianKernel(w, w/2)
        Y_resultant_kernel = cv2.getGaussianKernel(h, h/2)
        kernel_2d = Y_resultant_kernel * X_resultant_kernel.T
        mask = kernel_2d / kernel_2d.max()
        
        img = img.astype(float)
        for i in range(3):
            img[:, :, i] = img[:, :, i] * (0.85 + 0.15 * mask)
            
        return np.clip(img, 0, 255).astype(np.uint8)

    def synthesize_photo(self, doc_layer, output_path):
        """
        Projects doc_layer onto desk and applies mobile artifacts.
        doc_layer: RGBA image of the sovereign document.
        """
        desk = self.desk_img.copy()
        h_d, w_d = desk.shape[:2]
        h_i, w_i = doc_layer.shape[:2]
        
        # 1. Define Realistic Phone-Capture Perspective
        # (Top-down but slightly tilted and offset)
        src_pts = np.float32([[0, 0], [w_i, 0], [w_i, h_i], [0, h_i]])
        
        # Center the paper on the desk with a slight tilt
        cx, cy = w_d // 2, h_d // 2
        dw, dh = w_d // 3.5, h_d // 1.5
        dst_pts = np.float32([
            [cx - dw + 20, cy - dh + 40],   # TL
            [cx + dw - 10, cy - dh - 20],   # TR
            [cx + dw + 50, cy + dh - 10],   # BR
            [cx - dw - 30, cy + dh + 30]    # BL
        ])
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_doc = cv2.warpPerspective(doc_layer, M, (w_d, h_d), flags=cv2.INTER_LANCZOS4)
        
        # 2. Shadows and Blending
        alpha = (warped_doc[:, :, 3] / 255.0)[:, :, None]
        desk = self.add_contact_shadow(desk, warped_doc[:, :, 3])
        
        # Blend Document
        res = desk.astype(float) * (1.0 - alpha) + warped_doc[:, :, :3].astype(float) * alpha
        res = np.clip(res, 0, 255).astype(np.uint8)
        
        # 3. Mobile Post-Process
        res = self.apply_mobile_distortions(res)
        
        # Add high-ISO sensor noise (fine-grain)
        noise = np.random.normal(0, 4.2, res.shape).astype(np.float32)
        res = np.clip(res.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        cv2.imwrite(str(output_path), res, [cv2.IMWRITE_JPEG_QUALITY, 93])
        return res

if __name__ == "__main__":
    # Test stub
    pass
