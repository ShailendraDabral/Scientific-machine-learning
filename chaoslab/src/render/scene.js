import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export const COLORS = {
  amber: 0xe8a33d, cyan: 0x4cc9f0, rose: 0xf06292, green: 0x7bd88f,
  line: 0x2f3947, faint: 0x1c232d, text: 0x8b97a6,
};

/** Owns the WebGL renderer, camera and orbit controls for one viewport. */
export class SceneManager {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    container.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 200);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(container);
    this.resize();
  }

  setCamera(position, target = [0, 0, 0]) {
    this.camera.position.set(...position);
    this.controls.target.set(...target);
    this.controls.update();
  }

  resize() {
    const w = this.container.clientWidth || 1, h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.observer.disconnect();
    this.controls.dispose();
    this.scene.traverse((o) => { o.geometry?.dispose(); o.material?.dispose(); });
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}

/** Polyline with a fixed vertex budget; the oldest quarter is dropped when full. */
export class Trail {
  constructor(capacity, color, opacity = 0.9) {
    this.capacity = capacity;
    this.count = 0;
    this.positions = new Float32Array(capacity * 3);
    this.geometry = new THREE.BufferGeometry();
    this.attr = new THREE.BufferAttribute(this.positions, 3);
    this.attr.setUsage(THREE.DynamicDrawUsage);
    this.geometry.setAttribute('position', this.attr);
    this.geometry.setDrawRange(0, 0);
    this.object = new THREE.Line(this.geometry,
      new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
    this.object.frustumCulled = false;
    this.head = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), new THREE.MeshBasicMaterial({ color }));
    this.head.scale.setScalar(0.018);
    this.head.visible = false;
  }

  addTo(parent) { parent.add(this.object, this.head); return this; }

  push(x, y, z) {
    if (this.count === this.capacity) {
      const drop = this.capacity >> 2;
      this.positions.copyWithin(0, drop * 3, this.count * 3);
      this.count -= drop;
    }
    const i = this.count * 3;
    this.positions[i] = x; this.positions[i + 1] = y; this.positions[i + 2] = z;
    this.count++;
  }

  commit() {
    this.geometry.setDrawRange(0, this.count);
    this.attr.needsUpdate = true;
    if (this.count) {
      const i = (this.count - 1) * 3;
      this.head.position.set(this.positions[i], this.positions[i + 1], this.positions[i + 2]);
      this.head.visible = true;
    }
  }

  clear() { this.count = 0; this.head.visible = false; this.geometry.setDrawRange(0, 0); }
}

/** Static line segments from a flat list of points [x0,y0,z0, x1,y1,z1, ...] (pairs). */
export function segments(points, color = COLORS.line, opacity = 1) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
  return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity }));
}

/** Wireframe box between two corners, as line segments. */
export function boxFrame(min, max, color = COLORS.faint) {
  const [a, b, c] = min, [d, e, f] = max, p = [];
  const E = [[a, b, c, d, b, c], [a, e, c, d, e, c], [a, b, f, d, b, f], [a, e, f, d, e, f],
    [a, b, c, a, e, c], [d, b, c, d, e, c], [a, b, f, a, e, f], [d, b, f, d, e, f],
    [a, b, c, a, b, f], [d, b, c, d, b, f], [a, e, c, a, e, f], [d, e, c, d, e, f]];
  for (const s of E) p.push(...s);
  return segments(p, color);
}

export function sphere(color, radius) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), new THREE.MeshBasicMaterial({ color }));
  m.scale.setScalar(radius);
  return m;
}

/** Growing point cloud (for Poincare sections). */
export class PointSet {
  constructor(capacity, color, size = 3) {
    this.capacity = capacity; this.count = 0;
    this.positions = new Float32Array(capacity * 3);
    this.geometry = new THREE.BufferGeometry();
    this.attr = new THREE.BufferAttribute(this.positions, 3);
    this.geometry.setAttribute('position', this.attr);
    this.geometry.setDrawRange(0, 0);
    this.object = new THREE.Points(this.geometry,
      new THREE.PointsMaterial({ color, size, sizeAttenuation: false }));
    this.object.frustumCulled = false;
  }
  push(x, y, z) {
    const i = (this.count % this.capacity) * 3;
    this.positions[i] = x; this.positions[i + 1] = y; this.positions[i + 2] = z;
    this.count++;
    this.geometry.setDrawRange(0, Math.min(this.count, this.capacity));
    this.attr.needsUpdate = true;
  }
  clear() { this.count = 0; this.geometry.setDrawRange(0, 0); }
}
