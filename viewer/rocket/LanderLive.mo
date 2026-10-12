model LanderLive
  "ALSETLab RocketLanding.SimpleLander.rocket as a 1-D point-mass lander with a live throttle input and the CasADi hoverslam autopilot, for Rumoca's interactive 3-D viewer. Self-contained: no package dependencies."
  // --- live inputs, driven by the Rumoca scenario ([signals.model_inputs]) ---
  input Real throttle_cmd(min=0, max=1, start=0) "Manual throttle command, 0..1 (W/S keys)";
  input Real autopilot(start=1) "1 = CasADi hoverslam schedule, 0 = manual throttle (Space toggles)";
  // --- parameters carried over from RocketLanding.SimpleLander.rocket ---
  parameter Real EmptyRocket(unit="kg") = 100 "Dry mass";
  parameter Real FuelBurnRate(unit="kg/s") = 10 "Propellant flow at full throttle";
  parameter Real Fengine(unit="N") = 2500 "Thrust at full throttle";
  parameter Real rho(unit="kg/m3") = 1.225 "Air density";
  parameter Real S(unit="m2") = 1.632 "Reference area";
  parameter Real cd = 0.2 "Drag coefficient";
  parameter Real g(unit="m/s2") = 9.81 "Gravity";
  parameter Real eps(unit="m/s") = 0.1 "Smoothing of |v| in the drag law";
  parameter Real y_ign(unit="m") = 402.45 "Autopilot ignition altitude. The CasADi minimum-propellant hoverslam switches at t = 4.98775 s, i.e. at 402.51 m on the coast (87.1 m/s down); igniting 6 cm lower gives a firm ~1 m/s touchdown instead of a zero-speed graze. Triggered on altitude, not time, so the R reset (which restores the states but not the clock) works.";
  parameter Real v_crash(unit="m/s") = 3.0 "Touchdown speed above which the landing counts as a crash";
  // --- states ---
  Real y(unit="m", start=800, fixed=true) "Altitude above the pad";
  Real v(unit="m/s", start=-70, fixed=true) "Vertical velocity, up positive";
  Real fuel_mass(unit="kg", start=100, fixed=true) "Propellant mass";
  // --- discrete state latched at touchdown ---
  discrete Real landed(start=0, fixed=true) "1 once the vehicle has touched the pad";
  discrete Real ignited(start=0, fixed=true) "1 once the autopilot has lit the engine";
  discrete Real touchdown_speed(unit="m/s", start=0, fixed=true) "Vertical velocity at contact";
  // --- algebraic / outputs for the viewer ---
  Real u "Effective throttle 0..1";
  Real thrust(unit="N") "Engine thrust";
  Real Fdrag(unit="N") "Aerodynamic drag (opposes v)";
  Real crashed "1 if landed with |touchdown_speed| > v_crash";
  output Real position[3](each unit="m") "FLU position of the vehicle base {0, 0, max(y,0)}";
  output Real quat[4] "Body-to-world quaternion, scalar first (always identity)";
  output Real vx(unit="m/s") "Forward speed (zero)";
  output Real vy(unit="m/s") "Left speed (zero)";
  output Real vz(unit="m/s") "Up speed (= v)";
equation
  u = if landed > 0.5 then 0
      else if autopilot > 0.5 then (if ignited > 0.5 and fuel_mass > 0 then 1 else 0)
      else (if fuel_mass > 0 then throttle_cmd else 0);
  thrust = u*Fengine;
  // Same law as RocketLanding (0.5*rho*v^2*S*cd opposing the motion), written smooth: v*sqrt(v^2+eps^2) ~ v*|v|
  Fdrag = 0.5*rho*S*cd*v*sqrt(v*v + eps*eps);
  der(y) = if landed > 0.5 then 0 else v;
  der(v) = if landed > 0.5 then 0 else (thrust - Fdrag)/(EmptyRocket + fuel_mass) - g;
  der(fuel_mass) = -u*FuelBurnRate;
  when y <= y_ign then
    ignited = 1;
  end when;
  when y <= 0 then
    landed = 1;
    touchdown_speed = pre(v);
    reinit(v, 0);
  end when;
  crashed = if landed > 0.5 and abs(touchdown_speed) > v_crash then 1 else 0;
  position = {0, 0, max(y, 0)};
  quat = {1, 0, 0, 0};
  vx = 0;
  vy = 0;
  vz = v;
end LanderLive;
