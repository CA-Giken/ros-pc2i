from ros_pc2i.base.material import Material

class Metal(Material):
    def __init__(self, color=(0.95, 0.93, 0.88), ambient=0.2, diffuse=0.7, specular=0.8, shininess=64.0, metallic=0.9, roughness=0.1):
        super().__init__(color, ambient, diffuse, specular, shininess, metallic, roughness)