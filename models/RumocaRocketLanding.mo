within ;
package RumocaRocketLanding
  "ALSETLab RocketLanding.SimpleLander.rocket (Canham, Podlaski, Vanfretti, Modelica 2021) re-cast for Rumoca JAX/CasADi export"

  model Lander
    "Point-mass retrograde landing burn with a throttle input. Pure explicit ODE: Rumoca 0.10.2's casadi-ode/jax-ode targets reject algebraic variables and initial-equation sections, so forces are inlined and initial conditions use start/fixed."
    // Parameters carried over from RocketLanding.SimpleLander.rocket (units corrected to kg/s and kg/m3)
    parameter Real EmptyRocket(unit="kg") = 100 "Dry mass";
    parameter Real FuelBurnRate(unit="kg/s") = 10 "Propellant flow at full throttle";
    parameter Real Fengine(unit="N") = 2500 "Thrust at full throttle";
    parameter Real rho(unit="kg/m3") = 1.225 "Air density";
    parameter Real S(unit="m2") = 1.632 "Reference area";
    parameter Real cd = 0.2 "Drag coefficient";
    parameter Real g(unit="m/s2") = 9.81 "Gravity";
    parameter Real eps(unit="m/s") = 0.1 "Smoothing of |v| in the drag law";
    input Real u(min=0, max=1, start=0) "Throttle command, 0..1";
    Real y(unit="m", start=800, fixed=true) "Altitude above the pad";
    Real v(unit="m/s", start=-70, fixed=true) "Vertical velocity, up positive";
    Real fuel_mass(unit="kg", start=100, fixed=true) "Propellant mass";
  equation
    der(y) = v;
    // Original: Fdrag = if Ascending then -cd*qbar*S else cd*qbar*S with qbar = 0.5*rho*v^2.
    // Same law written smooth (v*sqrt(v^2+eps^2) ~ v*|v|) so JAX/CasADi can differentiate through it.
    der(v) = (u*Fengine - 0.5*rho*S*cd*v*sqrt(v*v + eps*eps))/(EmptyRocket + fuel_mass) - g;
    der(fuel_mass) = -u*FuelBurnRate;
  end Lander;

  model LanderBurn
    "Lander flown open loop for Rumoca's native simulator and OpenModelica: coast, then constant throttle from t_ign (the hoverslam schedule found with CasADi) until the tank is empty. Terminates at touchdown."
    parameter Real t_ign(unit="s") = 3.0 "Ignition time";
    parameter Real u_burn = 1.0 "Throttle after ignition";
    Lander lander;
    Real Fthrust(unit="N") "Engine thrust";
    Real rocket_mass(unit="kg") "Total mass";
  equation
    // Engine cut-off when the tank is empty, as EngineOn = fuel_mass > 0 in the original model
    lander.u = if time >= t_ign and lander.fuel_mass > 0 then u_burn else 0;
    Fthrust = lander.u*lander.Fengine;
    rocket_mass = lander.EmptyRocket + lander.fuel_mass;
    when lander.y <= 0 then
      terminate("touchdown");
    end when;
  end LanderBurn;

end RumocaRocketLanding;
