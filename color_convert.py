import numpy as np


# RGB -> XYZ
def RGBtoXYZ_convert(image_rgb: np.ndarray) -> np.ndarray:
    rgb = image_rgb.astype(np.float32) / 255.0
    var_R = rgb[:, :, 0]
    var_G = rgb[:, :, 1]
    var_B = rgb[:, :, 2]
    
    def gamma_correct(c):
        return np.where(
            c <= 0.04045,
            c / 12.92,
            ((c + 0.055) / 1.055) ** 2.4,
        )
    
    var_R = gamma_correct(var_R)
    var_G = gamma_correct(var_G)
    var_B = gamma_correct(var_B)
    
    var_R *= 100.0
    var_G *= 100.0
    var_B *= 100.0
    
    X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
    Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
    Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505
    
    image_xyz = np.zeros_like(image_rgb, dtype=np.float32)
    image_xyz[:, :, 0] = X
    image_xyz[:, :, 1] = Y
    image_xyz[:, :, 2] = Z
    return image_xyz

# XYZ -> Lab
def XYZtoLab_convert(image_xyz: np.ndarray) -> np.ndarray:
    X = image_xyz[:, :, 0]
    Y = image_xyz[:, :, 1]
    Z = image_xyz[:, :, 2]

    # D65 / 2° の参照光
    ref_X = 95.047
    ref_Y = 100.000
    ref_Z = 108.883

    var_X = X / ref_X
    var_Y = Y / ref_Y
    var_Z = Z / ref_Z

    def f(t):
        return np.where(
            t > 0.008856,
            t ** (1.0 / 3.0),
            7.787 * t + 16.0 / 116.0,
        )

    fX = f(var_X)
    fY = f(var_Y)
    fZ = f(var_Z)

    l = 116.0 * fY - 16.0
    a = 500.0 * (fX - fY)
    b = 200.0 * (fY - fZ)

    image_lab = np.zeros_like(image_xyz, dtype=np.float32)
    image_lab[:, :, 0] = l
    image_lab[:, :, 1] = a
    image_lab[:, :, 2] = b
    
    return image_lab

# Lab -> XYZ
def LabtoXYZ_convert(image_lab: np.ndarray) -> np.ndarray:
    L = image_lab[:, :, 0]
    a = image_lab[:, :, 1]
    b = image_lab[:, :, 2]

    # D65 / 2°
    ref_X = 95.047
    ref_Y = 100.000
    ref_Z = 108.883

    fY = (L + 16.0) / 116.0
    fX = a / 500.0 + fY
    fZ = fY - b / 200.0

    def finv(t):
        return np.where(
            t ** 3 > 0.008856,
            t ** 3,
            (t - 16.0 / 116.0) / 7.787,
        )

    var_Y = finv(fY)
    var_X = finv(fX)
    var_Z = finv(fZ)

    X = ref_X * var_X
    Y = ref_Y * var_Y
    Z = ref_Z * var_Z

    image_xyz = np.zeros_like(image_lab, dtype=np.float32)
    image_xyz[:, :, 0] = X
    image_xyz[:, :, 1] = Y
    image_xyz[:, :, 2] = Z
    return image_xyz

# XYZ -> RGB
def XYZtoRGB_convert(image_xyz: np.ndarray) -> np.ndarray:
    X = image_xyz[:, :, 0] / 100.0
    Y = image_xyz[:, :, 1] / 100.0
    Z = image_xyz[:, :, 2] / 100.0

    var_R = X * 3.2406 + Y * -1.5372 + Z * -0.4986
    var_G = X * -0.9689 + Y * 1.8758 + Z * 0.0415
    var_B = X * 0.0557 + Y * -0.2040 + Z * 1.0570

    def gamma_correct(c):
        return np.where(
            c > 0.0031308,
            1.055 * (c ** (1.0 / 2.4)) - 0.055,
            12.92 * c,
        )

    var_R = gamma_correct(var_R)
    var_G = gamma_correct(var_G)
    var_B = gamma_correct(var_B)

    R = np.clip(var_R * 255.0, 0, 255).astype(np.uint8)
    G = np.clip(var_G * 255.0, 0, 255).astype(np.uint8)
    B = np.clip(var_B * 255.0, 0, 255).astype(np.uint8)

    image_rgb = np.zeros_like(image_xyz, dtype=np.uint8)
    image_rgb[:, :, 0] = R
    image_rgb[:, :, 1] = G
    image_rgb[:, :, 2] = B
    return image_rgb