"use strict";

/* Given a `m` * `n` matrix `a`, and a `m`-element vector `b`, solve the linear
 * system `a` * `out` = `b`, and write the result to the output parameter `out`.
 *
 * The particular method to solve the linear system is to first compute the
 * singular value decomposition (or SVD) of `a`. The SVD of `a` can be used to
 * solve the linear system, by using it to compute the pseudoinverse of `a`.
 * If `a` is invertible, the pseudoinverse of `a` is equivalent to its inverse.
 * Otherwise, it computes a best-fit solution that minimizes the distance to
 * each plane. If this solution is not uniquely defined, it computes the unique
 * solution that minimizes the distance to the origin.
 *
 * To improve the numerical stability of the algorithm, we truncate singular
 * values smaller than a certain treshold.
 */
function solve(out, m, n, a, b) {
  // GSL requires that m >= n.
  var mext = Math.max(n, m);

  // Copy the matrix `a` to a GSL matrix.
  var acopy = _gsl_matrix_alloc(mext, n);
  for (var i = 0; i < m; i += 1)
  for (var j = 0; j < n; j += 1) {
    _gsl_matrix_set(acopy, i, j, a[3 * i + j]);
  }
  for (var i = m; i < mext; i += 1) {
    _gsl_matrix_set(acopy, i, 0, 0);
    _gsl_matrix_set(acopy, i, 1, 0);
    _gsl_matrix_set(acopy, i, 2, 0);
  }

  // Copy the vector `b` to A GSL VECTOR.
  var bcopy = _gsl_vector_alloc(mext);
  for (var i = 0; i < m; i += 1) {
    _gsl_vector_set(bcopy, i, b[i]);
  }
  for (var i = m; i < mext; i += 1) {
    _gsl_vector_set(bcopy, i, 0);
  }

  // Compute the SVD.
  var v = _gsl_matrix_alloc(n, n);
  var s = _gsl_vector_alloc(n);
  var work = _gsl_vector_alloc(3);
  _gsl_linalg_SV_decomp(acopy, v, s, work);
  _gsl_vector_free(work);


  // Truncate small singular values.
  for (var j = 0; j < n; j += 1) {
    var sj = _gsl_vector_get(s, j);
    _gsl_vector_set(s, j, sj < 0.1 ? 0.0 : sj);
  }

  // Solve the linear system.
  var outcopy = _gsl_vector_alloc(n);
  _gsl_linalg_SV_solve(acopy, v, s, bcopy, outcopy);
  _gsl_vector_free(s);
  _gsl_matrix_free(v);
  _gsl_matrix_free(bcopy);
  _gsl_matrix_free(acopy);

  // Write the result to the output vector `out`.
  for (var j = 0; j < n; j += 1) {
    out[j] = _gsl_vector_get(outcopy, j);
  }
  _gsl_vector_free(outcopy);
  return out;
}

/* Given a sample grid with minimum point `pmin`, maximum point `pmax`,
 * dimensions given by the vector `size`, and a volume represented by a distance
 * function `distance`, compute a set of vertices and triangle indices that
 * represent an isosurface for the given volume.
 */
function generateIsosurface(pmin, pmax, size, distance) {
  /* Given a vertex with coordinates `i`, `j`, and `k`, returns the index of the
   * vertex in an array of (m + 1) * (n + 1) * (p + 1) vertices.
   */
  function vertexIndex(i, j, k) {
    return (p + 1) * ((n + 1) * i + j) + k;
  }

  /* Given an edge in the (positive) x-direction with coordinates `i`, `j`, and
   * `k`, returns the index of the edge in an array of m * (n + 1) * (p + 1)
   * edges.
   */
  function edgeIndexX(i, j, k) {
    return (p + 1) * ((n + 1) * i + j) + k;
  }

  /* Given an edge in the (positive) y-direction with coordinates `i`, `j`, and
   * `k`, returns the index of the edge in an array of (m + 1) * n * (p + 1)
   * edges.
   */
  function edgeIndexY(i, j, k) {
    return (p + 1) * (n * i + j) + k;
  }

  /* Given an edge in the (positive) y-direction with coordinates `i`, `j`, and
   * `k`, returns the index of the edge in an array of (m + 1) * (n + 1) * p
   * edges.
  */
  function edgeIndexZ(i, j, k) {
    return p * ((n + 1) * i + j) + k;
  }

  /* Given a cell with coordinates `i`, `j`, and `k`, returns the index of the
   * cell in an array of m * n * p cells.
   */
  function cellIndex(i, j, k) {
    return p * (n * i + j) + k;
  }

  /* Given an edge with endpoints given by the vertex indices `e0` and `e1`,
   * compute the intersection point of the edge with the volume, and write it to
   * the output parameter `out`.
   *
   * If the edge intersects the volume, returns the output parameter `out`.
   * Otherwise, returns `null` instead.
   */
  function intersectEdge(out, e0, e1) {
    // Let `d0` and `d1` be the distance at each endpoint.
    var d0 = ds[e0];
    var d1 = ds[e1];

    // If the sign of `d0` and `d1` are equivalent, the edge does not intersect
    // the volume.
    if (Math.sign(d0) === Math.sign(d1)) {
      return null;
    }

    // Compute the intersection point by linearly interpolating between `d0` and
    // `d1`.
    return vec3.lerp(out, ps[e0], ps[e1], Math.abs(d0) / Math.abs(d1 - d0));
  }

  /* Given a point `p`, compute the normal at that point, and write it to the
   * output parameter `out`. Returns the output parameter `out`.
   */
  function normal(out, p) {
    // Destructure the arguments.
    var x = p[0];
    var y = p[1];
    var z = p[2];

    // Compute the normal by partially differentiating the distance function,
    // and then normalizing the result.
    var h = 0.001;
    out[0] = (distance([x + h, y, z]) - distance([x - h, y, z])) / (2 * h);
    out[1] = (distance([x, y + h, z]) - distance([x, y - h, z])) / (2 * h);
    out[2] = (distance([x, y, z + h]) - distance([x, y, z - h])) / (2 * h);
    return vec3.normalize(out, out);
  }

  /* Given a cell with coordinates `i`, `j`, and `k`, generate a vertex that
   * best approximates the intersection point of the planes defined by the
   * intersection point and normal at that point of each edge of the cell that
   * intersects the volume, and write it to the output parameter `out`.
   *
   * If at least one edge of the cell intersects the volume, returns the output
   * parameter `out`. Otherwise, returns `null` instead.
   */
  function generateVertex(out, i, j, k) {
    // To generate a vertex, we set up a linear system Ax = b, where each row of
    // the matrix A contains the normal at the intersection point, and each
    // entry of the row vector b contains the dot product of the intersection
    // point and the normal at that point. Solving this linear system will give
    // us the desired vector.

    // We already precomputed the intersection points and normals at these
    // points for each edge, so we only need to gather this information for the
    // edges of the given cell into two separate arrays here.
    var qs = [];
    var ns = [];

    // Gather the intersection points and normals at these points for each edge
    // of the given cell that intersects the volume in the x-direction.
    var index = edgeIndexX(i, j, k);
    var q = qsx[index];
    if (q !== undefined) {
      qs.push(q);
      ns.push(nsx[index]);
    }

    var index = edgeIndexX(i, j, k + 1);
    var q = qsx[index];
    if (q !== undefined) {
      qs.push(q);
      ns.push(nsx[index]);
    }

    var index = edgeIndexX(i, j + 1, k);
    var q = qsx[index];
    if (q !== undefined) {
      qs.push(q);
      ns.push(nsx[index]);
    }

    var index = edgeIndexX(i, j + 1, k + 1);
    var q = qsx[index];
    if (q !== undefined) {
      qs.push(q);
      ns.push(nsx[index]);
    }

    // Gather the intersection points and normals at these points for each edge
    // of the given cell that intersects the volume in the y-direction.
    var index = edgeIndexY(i, j, k);
    var q = qsy[index];
    if (q !== undefined) {
      qs.push(q);
      ns.push(nsy[index]);
    }

    var index = edgeIndexY(i, j, k + 1);
    var q = qsy[index];
    if (q !== undefined) {
      qs.push(q);
      ns.push(nsy[index]);
    }

    var index = edgeIndexY(i + 1, j, k);
    var q = qsy[index];
    if (q !== undefined) {
      qs.push(q);
      ns.push(nsy[index]);
    }

    var index = edgeIndexY(i + 1, j, k + 1);
    var q = qsy[index];
    if (q !== undefined) {
      qs.push(q);
      ns.push(nsy[index]);
    }

    // Gather the intersection points and normals at these points for each edge
    // of the given cell that intersects the volume in the z-direction.
    var index = edgeIndexZ(i, j, k);
    var q = qsz[index];
    if (q !== undefined) {
      qs.push(q);
      ns.push(nsz[index]);
    }

    var index = edgeIndexZ(i, j + 1, k);
    var q = qsz[index];
    if (q !== undefined) {
      qs.push(q);
      ns.push(nsz[index]);
    }

    var index = edgeIndexZ(i + 1, j, k);
    var q = qsz[index];
    if (q !== undefined) {
      qs.push(q);
      ns.push(nsz[index]);
    }

    var index = edgeIndexZ(i + 1, j + 1, k);
    var q = qsz[index];
    if (q !== undefined) {
      qs.push(q);
      ns.push(nsz[index]);
    }

    // If the array is empty, no edge of the cell intersects the volume.
    var length = qs.length;
    if (length === 0) {
      return null;
    }

    // When the vertex that best approximates the intersection point of the
    // planes is not uniquely defined, we want the (unique) vertex that
    // minimizes the distance to the centroid of the intersection points.
    // To accomplish this, each intersection point is translated so that the
    // centroid is at the origin. After solving the linear system, we invert
    // this translation to obtain our final answer.

    // Compute the centroid of the intersection points.
    var c = vec3.create();
    for (var index = 0; index < length; index += 1) {
      vec3.add(c, c, qs[index]);
    }
    vec3.scale(c, c, 1 / length);

    // Create the linear aystem Ax = b.
    var a = new Array(length * 3);
    var b = new Array(length);
    var q = vec3.create();
    for (var index = 0; index < length; index += 1) {
      var ni = ns[index];
      a[3 * index] = ni[0];
      a[3 * index + 1] = ni[1];
      a[3 * index + 2] = ni[2];
      b[index] = vec3.dot(vec3.subtract(q, qs[index], c), ni);
    }

    // Solve the linear system Ax = b.
    return vec3.add(out, solve(out, length, 3, a, b), c);
  }

  function generateQuad(out, q00, q01, q10, q11) {
    out.push([q00, q10, q01]);
    out.push([q01, q10, q11]);
  }

  // Destructure the arguments.
  var xmin = pmin[0];
  var ymin = pmin[1];
  var zmin = pmin[2];
  var xmax = pmax[0];
  var ymax = pmax[1];
  var zmax = pmax[2];
  var m = size[0];
  var n = size[1];
  var p = size[2];

  // Sample the distance field.
  var ps = new Array((m + 1) * (n + 1) * (p + 1));
  var ds = new Array((m + 1) * (n + 1) * (p + 1));
  var dx = (xmax - xmin) / m;
  var dy = (ymax - ymin) / n;
  var dz = (zmax - zmin) / p;
  var x = xmin;
  for (var i = 0; i < m + 1; i += 1) {
    var y = ymin;
    for (var j = 0; j < n + 1; j += 1) {
      var z = zmin;
      for (var k = 0; k < p + 1; k += 1) {
        var index = vertexIndex(i, j, k);
        var q = vec3.fromValues(x, y, z);
        ps[index] = q;
        ds[index] = distance(q);
        z += dz;
      }
      y += dy;
    }
    x += dx;
  }

  // For each edge that intersects the volume in the x-direction, compute the
  // intersection point and normal at that point.
  var qsx = new Array(m * (n + 1) * (p + 1));
  var nsx = new Array(m * (n + 1) * (p + 1));
  for (var i = 0; i < m; i += 1) {
    for (var j = 0; j < n + 1; j += 1) {
      for (var k = 0; k < p + 1; k += 1) {
        var q = vec3.create();
        if (intersectEdge(
          q,
          vertexIndex(i, j, k),
          vertexIndex(i + 1, j, k)
        )) {
          var index = edgeIndexX(i, j, k);
          qsx[index] = q;
          nsx[index] = normal(vec3.create(), q);
        }
      }
    }
  }

  // For each edge that intersects the volume in the y-direction, compute the
  // intersection point and normal at that point.
  var qsy = new Array((m + 1) * n * (p + 1));
  var nsy = new Array((m + 1) * n * (p + 1));
  for (var i = 0; i < m + 1; i += 1) {
    for (var j = 0; j < n; j += 1) {
      for (var k = 0; k < p + 1; k += 1) {
        var q = vec3.create();
        if (intersectEdge(
          q,
          vertexIndex(i, j, k),
          vertexIndex(i, j + 1, k)
        )) {
          var index = edgeIndexY(i, j, k);
          qsy[index] = q;
          nsy[index] = normal(vec3.create(), q);
        }
      }
    }
  }

  // For each edge that intersects the volume in the z-direction, compute the
  // intersection point and normal at that point.
  var qsz = new Array((m + 1) * (n + 1) * p);
  var nsz = new Array((m + 1) * (n + 1) * p);
  for (var i = 0; i < m + 1; i += 1) {
    for (var j = 0; j < n + 1; j += 1) {
      for (var k = 0; k < p; k += 1) {
        var q = vec3.create();
        if (intersectEdge(
          q,
          vertexIndex(i, j, k),
          vertexIndex(i, j, k + 1)
        )) {
          var index = edgeIndexZ(i, j, k);
          qsz[index] = q;
          nsz[index] = normal(vec3.create(), q);
        }
      }
    }
  }

  // For each cell that contains at least one edge that intersects the volume,
  // generate a vertex that best approximates the intersection of the planes
  // defined by the intersection point and normal at that point of each edge.
  var vs = new Array(m * n * p);
  for (var i = 0; i < m; i += 1)
  for (var j = 0; j < n; j += 1)
  for (var k = 0; k < p; k += 1) {
    var v = vec3.create();
    if (generateVertex(v, i, j, k)) {
      var index = cellIndex(i, j, k);
      vs[index] = v;
    }
  }

  var ts = [];

  // For each quadruple of cells that share an edge that intersects the volume
  // in the x-direction, generate a quad that connects the vertices of each
  // cell.
  for (var i = 0; i < m; i += 1)
  for (var j = 0; j < n - 1; j += 1)
  for (var k = 0; k < p - 1; k += 1) {
    var index = edgeIndexX(i, j + 1, k + 1);
    if (qsx[index] !== undefined) {
      generateQuad(
        ts,
        cellIndex(i, j, k),
        cellIndex(i, j + 1, k),
        cellIndex(i, j, k + 1),
        cellIndex(i, j + 1, k + 1)
      );
    }
  }

  // For each quadruple of cells that share an edge that intersects the volume
  // in the y-direction, generate a quad that connects the vertices of each
  // cell.
  for (var j = 0; j < n; j += 1)
  for (var i = 0; i < m - 1; i += 1)
  for (var k = 0; k < p - 1; k += 1) {
    var index = edgeIndexY(i + 1, j, k + 1);
    if (qsy[index] !== undefined) {
      generateQuad(
        ts,
        cellIndex(i, j, k),
        cellIndex(i, j, k + 1),
        cellIndex(i + 1, j, k),
        cellIndex(i + 1, j, k + 1)
      );
    }
  }

  // For each quadruple of cells that share an edge that intersects the volume
  // in the z-direction, generate a quad that connects the vertices of each
  // cell.
  for (var k = 0; k < p; k += 1)
  for (var i = 0; i < m - 1; i += 1)
  for (var j = 0; j < n - 1; j += 1) {
    var index = edgeIndexZ(i + 1, j + 1, k);
    if (qsz[index] !== undefined) {
      generateQuad(
        ts,
        cellIndex(i, j, k),
        cellIndex(i, j + 1, k),
        cellIndex(i + 1, j, k),
        cellIndex(i + 1, j + 1, k)
      );
    }
  }

  return [vs, ts];
}

// Returns a distance function for a box of size 1, centered at the origin.
function box() {
  var ones = vec3.fromValues(1, 1, 1);
  var zeroes = vec3.create();

  return function (p) {
    var d = vec3.fromValues(Math.abs(p[0]), Math.abs(p[1]), Math.abs(p[2]));
    vec3.subtract(d, d, ones);

    return Math.min(Math.max(d[0], d[1], d[2]), 0.0) +
          vec3.length(vec3.max(vec3.create(), zeroes, d));
  };
}

// Returns a distance function for a cylinder of radius 1 and height 1, centered
// at the origin.
function cylinder() {
  var ones = vec2.fromValues(1, 1);
  var zeroes = vec2.create();

  return function (p) {
    var d = vec2.fromValues(vec2.length(vec2.fromValues(p[0], p[2])), p[1]);
    d[0] = Math.abs(d[0]);
    d[1] = Math.abs(d[1]);
    vec2.subtract(d, d, ones);

    return Math.min(Math.max(d[0], d[1]), 0.0) +
           vec2.length(vec2.max(vec2.create(), zeroes, d));
  };
}

// Returns a distance function for a sphere of radius 1, centered at the origin.
function sphere() {
  return function (p) {
    return vec3.length(p) - 1;
  };
}

// Given a distance function `distance` and an angle `rx`, returns a distance
// function that represents the same volume rotated by an angle `rx` around the
// x-axis.
function rotateX(distance, rx) {
  var zeroes = vec3.create();

  return function (p) {
    return distance(vec3.rotateX(vec3.create(), p, zeroes, -rx));
  };
}

// Given a distance function `distance` and an angle `ry`, returns a distance
// function that represents the same volume rotated by an angle `ry` around the
// y-axis.
function rotateY(distance, ry) {
  var zeroes = vec3.create();

  return function (p) {
    return distance(vec3.rotateY(vec3.create(), p, zeroes, -ry));
  };
}

// Given a distance function `distance` and an angle `rz`, returns a distance
// function that represents the same volume rotated by an angle `rz` around the
// z-axis.
function rotateZ(distance, rz) {
  var zeroes = vec3.create();

  return function (p) {
    return distance(vec3.rotateZ(vec3.create(), p, zeroes, -rz));
  };
}

// Given a distance function `distance` and a scaling vector `s`, returns a
// distance function that represents the same volume scaled by the scaling
// vector `s`.
function scale(distance, s) {
  vec3.inverse(s, s);

  return function (p) {
    return distance(vec3.multiply(vec3.create(), p, s));
  };
}

// Given a distance function `distance` and a translation vector `s`, returns a
// distance function that represents the same volume translated by the
// translation vector `t`.
function translate(distance, t) {
  vec3.negate(t, 0, t, 0);

  return function (p) {
    return distance(vec3.add(vec3.create(), p, t));
  };
}

// Given two distance functions `distance1` and `distance2`, returns a distance
// function that represents the difference of these two distance functions.
function difference(distance1, distance2) {
  return function (p) {
    return Math.max(distance1(p), -distance2(p));
  };
}

// Given two distance functions `distance1` and `distance2`, returns a distance
// function that represents the intersection of these two distance functions.
function intersection(distance1, distance2) {
  return function (p) {
    return Math.max(distance1(p), distance2(p));
  };
}

// Given two distance functions `distance1` and `distance2`, returns a distance
// function that represents the union of these two distance functions.
function union(distance1, distance2) {
  return function (p) {
    return Math.min(distance1(p), distance2(p));
  };
}

var vertexShader = [
  "uniform mat4 uMatrix;",
  "",
  "attribute vec3 aVertex;",
  "",
  "varying vec3 vVertex;",
  "",
  "void main(void) {",
  "  vec4 position = vec4(aVertex, 1.0);",
  "  gl_Position = uMatrix * position;",
  "  vVertex = gl_Position.xyz;",
  "}"
].join("\n");

var fragmentShader = [
  "#extension GL_OES_standard_derivatives : enable",
  "",
  "precision mediump float;",
  "",
  "varying vec3 vVertex;",
  "",
  "void main(void) {",
  "  vec3 dx = dFdx(vVertex);",
  "  vec3 dy = dFdy(vVertex);",
  "  vec3 normal = normalize(cross(dx, dy));",
  "  vec3 light = vec3(1.0);",
  "  float diffuse = dot(normal, -light);",
  "  gl_FragColor = vec4(vec3(0.25 * diffuse + 0.5), 1.0);",
  "}"
].join("\n");

window.onload = function () {
  function compileShader(type, source) {
    var shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader));
    }
    return shader;
  }

  function renderFrame() {
    var m = mat4.create();
    mat4.identity(m);
    mat4.scale(m, m, [0.5, 0.5, 0.5]);
    mat4.rotate(m, m, t, [1.0, 1.0, 1.0]);

    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(program);
    gl.uniformMatrix4fv(uMatrix, false, m);
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.vertexAttribPointer(aVertex, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.drawElements(gl.TRIANGLES, indexCount, gl.UNSIGNED_SHORT, 0);
    t += 0.01;
    requestAnimationFrame(renderFrame);
  }

  function generateMesh() {
    // Compute the isosurface.
    var now = new Date();
    var result = generateIsosurface(
      [-1, -1, -1],
      [+1, +1, +1],
      [32, 32, 32],
      eval(textArea.value)
    );
    var vs = result[0];
    var ts = result[1];
    console.log(new Date() - now);

    // Flatten the vertex array.
    var length = vs.length;
    var vertices = new Float32Array(length * 3);
    for (var index = 0; index < length; index += 1) {
      var v = vs[index];
      if (v !== undefined) {
        vertices[3 * index] = v[0];
        vertices[3 * index + 1] = v[1];
        vertices[3 * index + 2] = v[2];
      }
    }

    // Flatten the index array.
    var length = ts.length;
    var indices = new Uint16Array(length * 3);
    for (var index = 0; index < length; index += 1) {
      var t = ts[index];
      indices[3 * index] = t[0];
      indices[3 * index + 1] = t[1];
      indices[3 * index + 2] = t[2];
    }

    // Create the vertex buffer.
    vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

    // Create the index buffer.
    indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices),
                  gl.STATIC_DRAW);

    // Compute the index count.
    indexCount = 3 * length;
  }

  // Initialize WebGL.
  var canvas = document.getElementById("canvas");
  var gl = canvas.getContext("webgl");
  gl.getExtension("OES_standard_derivatives");
  gl.clearColor(0.0, 0.0, 0.0, 1.0);
  gl.enable(gl.DEPTH_TEST);

  // Compile and link the shaders.
  var program = gl.createProgram();
  gl.attachShader(program, compileShader(gl.VERTEX_SHADER, vertexShader));
  gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, fragmentShader));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program));
  }

  // Initialize the uniforms and attributes.
  var uMatrix = gl.getUniformLocation(program, "uMatrix");
  var aVertex = gl.getAttribLocation(program, "aVertex");
  gl.enableVertexAttribArray(aVertex);

  // Generate the initial mesh.
  var textArea = document.getElementById("textarea");
  var vertexBuffer;
  var indexBuffer;
  var indexCount;
  generateMesh();
  window.generateMesh = generateMesh;

  // Start the render loop.
  var t = 0;
  requestAnimationFrame(renderFrame);
};
