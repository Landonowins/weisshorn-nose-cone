"""Example plugin (user guide, section 7.1): the secant-ogive profile family.

Use it from a configuration file:

    project:
      plugins: [../plugins/secant_ogive.py]
    nose:
      custom_profiles:
        secant15: {family: secant_ogive, rho_factor: 1.5, label: "Secant ogive"}
      profiles: [ogive, secant15, vonkarman]
"""
import numpy as np

from nosecone.geometry import register_family


@register_family("secant_ogive", params={"rho_factor": 1.5}, dimensional=True,
                 doc="Secant ogive: circular arc of radius rho_factor x the tangent-ogive radius, "
                     "meeting the body at an angle (rho_factor = 1: tangent ogive)")
def secant_ogive(x, L, R, rho_factor):
    """Local radius y(x) [m] of a secant ogive of length L and base radius R."""
    rho = rho_factor * (R**2 + L**2) / (2 * R)
    alpha = np.arccos(np.sqrt(L**2 + R**2) / (2 * rho)) - np.arctan(R / L)
    x = np.clip(x, 0.0, L)
    return np.sqrt(np.maximum(rho**2 - (rho * np.cos(alpha) - x) ** 2, 0.0)) - rho * np.sin(alpha)
