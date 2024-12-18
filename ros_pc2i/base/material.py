from dataclasses import dataclass

@dataclass
class Material:
    """
    Material class
    """
    def __init__(self,
      color: tuple = (1.0, 1.0, 1.0),
      ambient: float = 0.1,
      diffuse: float = 0.7,
      specular: float = 0.2,
      shininess: float = 32.0,
      metallic: float = 0.9,
      roughness: float = 0.3
    ):
        """
        Parameters
        ----------
        color : tuple
            Material color
        ambient : float
            Ambient reflection coefficient
        diffuse : float
            Diffuse reflection coefficient
        specular : float
            Specular reflection coefficient
        shininess : float
            Shininess coefficient
        """
        self.color = color
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.metallic = metallic
        self.roughness = roughness
