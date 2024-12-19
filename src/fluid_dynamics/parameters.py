from pydantic import BaseModel, Field, validator


class FluidParameters(BaseModel):
    u_pt: float = Field(..., gt=0.0, description="Phase transition temperature [K].")
    u_ref: float = Field(..., gte=0.0, description="Reference temperature [K].")
    epsilon: float = Field(
        ..., gt=0.0, description="Parameter of the indicator function used in the fictitious domain method."
    )

    @property
    def kinematic_viscosity_at_u_ref(self) -> float:
        """
        Calculate the kinematic viscosity coefficient at the reference temperature.
        """
        return 5.56e-10 * self.u_ref * self.u_ref - 4.95e-8 * self.u_ref + 1.767e-6

    @property
    def u_pt_ref(self) -> float:
        """
        Calculate the deviation of phase transition temperature from the reference temperature.
        """
        return self.u_pt - self.u_ref

    def __str__(self):
        s = (
            f"Fluid Parameters:\n"
            f"  Phase Transition Temperature: {self.u_pt} K\n"
            f"  Reference Temperature: {self.u_ref} K\n"
            f"  Parameter of the Indicator Function (Epsilon): {self.epsilon}\n"
            f"  Kinematic Viscosity at the Reference Temperature: {self.kinematic_viscosity_at_u_ref} m^2/s\n"
        )
        return s
