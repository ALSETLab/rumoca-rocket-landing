model LanderLive
  "ALSETLab RocketLanding.SimpleLander.rocket as a 1-D point-mass lander with a live throttle input and the CasADi hoverslam autopilot, for Rumoca's interactive 3-D viewer. Self-contained: no package dependencies."
  // --- live inputs, driven by the Rumoca scenario ([signals.model_inputs]) ---
  input Real throttle_cmd(min=0, max=1, start=0) "Manual throttle command, 0..1 (W/S keys)";
  input Real autopilot(start=1) "1 = CasADi hoverslam schedule, 0 = manual throttle (Space toggles)";
  input Real loop_mode(start=1) "1 = after landing, wait hold seconds on the pad and fly again from the start (L toggles)";
  // --- parameters carried over from RocketLanding.SimpleLander.rocket ---
  parameter Real EmptyRocket(unit="kg") = 100 "Dry mass";
  parameter Real FuelBurnRate(unit="kg/s") = 10 "Propellant flow at full throttle";
  parameter Real Fengine(unit="N") = 2500 "Thrust at full throttle";
  parameter Real rho(unit="kg/m3") = 1.225 "Air density";
  parameter Real S(unit="m2") = 1.632 "Reference area";
  parameter Real cd = 0.2 "Drag coefficient";
  parameter Real g(unit="m/s2") = 9.81 "Gravity";
  parameter Real eps(unit="m/s") = 0.1 "Smoothing of |v| in the drag law";
  parameter Real y_ign(unit="m") = 402.40 "Autopilot ignition altitude. The CasADi minimum-propellant hoverslam switches at t = 4.98775 s, i.e. at 402.51 m on the coast (87.1 m/s down); igniting 11 cm lower gives a firm 1.8 m/s touchdown instead of a zero-speed graze, which Rumoca 0.10.2 can miss as a state event. Triggered on altitude, not time, so reset and loop restarts work.";
  parameter Real v_crash(unit="m/s") = 3.0 "Touchdown speed above which the landing counts as a crash";
  parameter Real hold(unit="s") = 4 "Time on the pad before the next flight in loop mode";
  parameter Real y_start(unit="m") = 800 "Start altitude of every flight";
  parameter Real v_start(unit="m/s") = -70 "Start vertical velocity of every flight";
  parameter Real fuel_start(unit="kg") = 100 "Propellant at the start of every flight";
  // --- states ---
  Real y(unit="m", start=y_start, fixed=true) "Altitude above the pad";
  Real v(unit="m/s", start=v_start, fixed=true) "Vertical velocity, up positive";
  Real fuel_mass(unit="kg", start=fuel_start, fixed=true) "Propellant mass";
  Real pad_timer(unit="s", start=0, fixed=true) "Time spent on the pad since touchdown (drives the loop restart as a state event)";
  // --- discrete state latched at touchdown ---
  discrete Real landed(start=0, fixed=true) "1 once the vehicle has touched the pad";
  discrete Real ignited(start=0, fixed=true) "1 once the autopilot has lit the engine";
  discrete Real touchdown_speed(unit="m/s", start=0, fixed=true) "Vertical velocity at contact";
  discrete Real t_land(unit="s", start=0, fixed=true) "Time of the last touchdown";
  discrete output Real flights(start=1, fixed=true) "Number of the current flight (counts loop restarts)";
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
  der(pad_timer) = if landed > 0.5 then 1 else 0;
  // Ignition, touchdown and loop restart share one when/elsewhen so each discrete variable has one definition.
  when y <= y_ign then
    ignited = 1;
    landed = pre(landed);
    touchdown_speed = pre(touchdown_speed);
    t_land = pre(t_land);
    flights = pre(flights);
  elsewhen y <= 0 then
    ignited = pre(ignited);
    landed = 1;
    touchdown_speed = pre(v);
    t_land = time;
    flights = pre(flights);
    reinit(v, 0);
  elsewhen loop_mode > 0.5 and pre(landed) > 0.5 and pad_timer >= hold then
    ignited = 0;
    landed = 0;
    touchdown_speed = pre(touchdown_speed);
    t_land = pre(t_land);
    flights = pre(flights) + 1;
    reinit(y, y_start);
    reinit(v, v_start);
    reinit(fuel_mass, fuel_start);
    reinit(pad_timer, 0);
  end when;
  crashed = if landed > 0.5 and abs(touchdown_speed) > v_crash then 1 else 0;
  position = {0, 0, max(y, 0)};
  quat = {1, 0, 0, 0};
  vx = 0;
  vy = 0;
  vz = v;
end LanderLive;
