// rocket_scene.js - ALSETLab RocketLanding hoverslam for the Rumoca sim viewer.
// Scene API: ctx.onInit(api), ctx.onFrame(api); api.THREE, api.scene, api.camera,
// api.cam {target, dist, angle, elev}, api.state, api.get(name), api.cameraMode.
// Model axes are Forward-Left-Up (Z up); Three.js (x3,y3,z3) = (-py, pz, px).

function num(v) { const n = Number(v); return Number.isFinite(n) ? n : 0; }

const FENGINE = 2500; // N at full throttle, same as the model parameter

ctx.onInit = async function (api) {
  const THREE = api.THREE, scene = api.scene, s = api.state;

  scene.background = new THREE.Color(0x8ec2ef);
  scene.fog = new THREE.Fog(0x8ec2ef, 300, 1500);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x4f6b3a, 0.95));
  const sun = new THREE.DirectionalLight(0xfff4e0, 1.1);
  sun.position.set(80, 160, 60);
  scene.add(sun);

  // Ground, grid and landing pad (pad centre is the model origin)
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(3000, 3000),
    new THREE.MeshStandardMaterial({ color: 0x6f9452, roughness: 1 }));
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.02;
  scene.add(ground);
  scene.add(new THREE.GridHelper(600, 60, 0x3d5a2a, 0x557a3c));
  const pad = new THREE.Mesh(new THREE.CylinderGeometry(7, 7.5, 0.3, 48),
    new THREE.MeshStandardMaterial({ color: 0x7a7f85, roughness: 0.8 }));
  pad.position.y = 0.15;
  scene.add(pad);
  const ring = new THREE.Mesh(new THREE.RingGeometry(5.0, 6.0, 64),
    new THREE.MeshBasicMaterial({ color: 0xffc400, side: THREE.DoubleSide }));
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.31;
  scene.add(ring);
  const dot = new THREE.Mesh(new THREE.CircleGeometry(0.8, 32),
    new THREE.MeshBasicMaterial({ color: 0xffc400, side: THREE.DoubleSide }));
  dot.rotation.x = -Math.PI / 2;
  dot.position.y = 0.31;
  scene.add(dot);

  // Rocket: group origin at the foot of the legs (= model altitude y)
  const rocket = new THREE.Group();
  const R = 0.3, H = 4.0, LEG = 1.0;
  s.bodyMat = new THREE.MeshStandardMaterial({ color: 0xc9ced3, metalness: 0.45, roughness: 0.4 });
  const body = new THREE.Mesh(new THREE.CylinderGeometry(R, R, H, 32), s.bodyMat);
  body.position.y = LEG * 0.6 + H / 2;
  rocket.add(body);
  const nose = new THREE.Mesh(new THREE.ConeGeometry(R, 1.0, 32), s.bodyMat);
  nose.position.y = LEG * 0.6 + H + 0.5;
  rocket.add(nose);
  const band = new THREE.Mesh(new THREE.CylinderGeometry(R + 0.01, R + 0.01, 0.4, 32),
    new THREE.MeshStandardMaterial({ color: 0x1f4e9c }));
  band.position.y = LEG * 0.6 + H - 0.6;
  rocket.add(band);
  const bell = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.24, 0.5, 24),
    new THREE.MeshStandardMaterial({ color: 0x2b2b2b, metalness: 0.7, roughness: 0.3 }));
  bell.position.y = LEG * 0.6 - 0.2;
  rocket.add(bell);
  const legMat = new THREE.MeshStandardMaterial({ color: 0x3a3a3a });
  const up = new THREE.Vector3(0, 1, 0);
  for (let i = 0; i < 4; i++) {
    const a = i * Math.PI / 2 + Math.PI / 4;
    const foot = new THREE.Vector3(Math.cos(a) * 0.85, 0, Math.sin(a) * 0.85);
    const top = new THREE.Vector3(Math.cos(a) * R, LEG, Math.sin(a) * R);
    const dir = top.clone().sub(foot);
    const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, dir.length(), 8), legMat);
    leg.position.copy(foot).add(top).multiplyScalar(0.5);
    leg.quaternion.setFromUnitVectors(up, dir.clone().normalize());
    rocket.add(leg);
    const padFoot = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.05, 12), legMat);
    padFoot.position.copy(foot).setY(0.025);
    rocket.add(padFoot);
  }
  // Exhaust: two cones (orange outer, yellow core) hanging from the bell, apex down; unit height scaled in onFrame
  s.flame = new THREE.Mesh(new THREE.ConeGeometry(0.26, 1, 24, 1, true),
    new THREE.MeshBasicMaterial({ color: 0xff7a1a, transparent: true, opacity: 0.8, depthWrite: false }));
  s.flame.rotation.x = Math.PI;
  rocket.add(s.flame);
  s.core = new THREE.Mesh(new THREE.ConeGeometry(0.14, 1, 24, 1, true),
    new THREE.MeshBasicMaterial({ color: 0xfff1a0, transparent: true, opacity: 0.9, depthWrite: false }));
  s.core.rotation.x = Math.PI;
  rocket.add(s.core);
  s.flameLight = new THREE.PointLight(0xff9a30, 0, 40);
  s.flameLight.position.y = -1;
  rocket.add(s.flameLight);
  s.bellY = LEG * 0.6 - 0.45;
  scene.add(rocket);
  s.rocket = rocket;

  // Altitude trail from pad to rocket, and a ground shadow disc
  s.trail = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 1, 6),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.3 }));
  scene.add(s.trail);
  s.shadow = new THREE.Mesh(new THREE.CircleGeometry(1.0, 32),
    new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.25, depthWrite: false }));
  s.shadow.rotation.x = -Math.PI / 2;
  s.shadow.position.y = 0.32;
  scene.add(s.shadow);

  if (api.cam) { api.cam.dist = 25; api.cam.elev = 0.12; api.cam.angle = Math.PI / 4; }
};

ctx.onFrame = function (api) {
  const s = api.state, get = api.get;
  if (!s.rocket) return;
  const px = num(get("px")), py = num(get("py")), pz = num(get("pz"));
  const tx = -py, ty = Math.max(pz, 0), tz = px;
  s.rocket.position.set(tx, ty, tz);

  // Exhaust length follows thrust (0 when the engine is off)
  const thrust = num(get("thrust"));
  const frac = Math.max(0, Math.min(thrust / FENGINE, 1));
  const L = frac > 0.01 ? 1.0 + 5.0 * frac + 0.4 * frac * Math.random() : 0;
  s.flame.visible = s.core.visible = L > 0;
  if (L > 0) {
    s.flame.scale.set(1, L, 1);
    s.flame.position.y = s.bellY - L / 2;
    s.core.scale.set(1, L * 0.65, 1);
    s.core.position.y = s.bellY - L * 0.65 / 2;
  }
  s.flameLight.intensity = 3 * frac;

  // Rocket colour: grey in flight, green when landed softly, red when crashed
  const landed = num(get("landed")) > 0.5, crashed = num(get("crashed")) > 0.5;
  s.bodyMat.color.setHex(landed ? (crashed ? 0xd62828 : 0x2fb344) : 0xc9ced3);
  s.bodyMat.emissive.setHex(landed ? (crashed ? 0x5a0000 : 0x0a3a10) : 0x000000);

  // Altitude trail and shadow
  s.trail.scale.set(1, Math.max(ty, 0.01), 1);
  s.trail.position.set(tx, ty / 2, tz);
  const sc = Math.max(0.3, 1.5 - ty * 0.0015);
  s.shadow.position.x = tx; s.shadow.position.z = tz;
  s.shadow.scale.set(sc, sc, 1);
  s.shadow.material.opacity = Math.max(0.04, 0.3 - ty * 0.0003);

  // Chase camera: orbit around the rocket, never below the ground
  if (api.cameraMode === "scene" && api.cam) {
    const c = api.cam;
    c.target.set(tx, ty + 2.5, tz);
    api.camera.position.set(
      c.target.x + c.dist * Math.sin(c.angle) * Math.cos(c.elev),
      c.target.y + c.dist * Math.sin(c.elev),
      c.target.z + c.dist * Math.cos(c.angle) * Math.cos(c.elev));
    api.camera.position.y = Math.max(api.camera.position.y, 1.5);
    api.camera.lookAt(c.target);
  }
};
