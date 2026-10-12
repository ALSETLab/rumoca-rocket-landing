model LanderBatchTest
  "Batch check of LanderLive with the autopilot engaged and no manual throttle (binds both inputs so rumoca sim can run it without a scenario)."
  LanderLive lander;
equation
  lander.throttle_cmd = 0;
  lander.autopilot = 1;
end LanderBatchTest;
