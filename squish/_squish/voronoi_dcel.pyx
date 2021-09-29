from _squish cimport SiteCacheMap, EdgeCacheMap, VoronoiInfo, Site, HalfEdge

#### Constants ####

init.SiteCacheMap, init.EdgeCacheMap, init.VoronoiInfo, init.Site, init.HalfEdge = \
	init_sitecachemap, init_edgecachemap, init_voronoiinfo, init_site, init_halfedge

cdef SiteCacheMap SITE_CACHE_MAP = init.SiteCacheMap(0, 1, 2, 3, 4)

cdef EdgeCacheMap AREA_EDGE_CACHE_MAP = init.EdgeCacheMap(0, 4, 6, 8, 10, -1, 12, 13,
															-1, -1, -1, -1, -1, 14)
cdef EdgeCacheMap RADIALT_EDGE_CACHE_MAP = init.EdgeCacheMap(0, 4, 6, 8, -1, 10, 12, 13,
																14, 15, 16, 17, 18, 19)

#### SiteCacheMap Methods ####

cdef inline SiteCacheMap init_sitecachemap(INT_T iarea, INT_T iperim, INT_T iisoparam, 
						INT_T ienergy, INT_T iavg_radius) nogil:
	cdef SiteCacheMap sc
	sc.iarea, sc.iperim, sc.iisoparam, sc.ienergy, sc.iavg_radius = \
		iarea, iperim, iisoparam, ienergy, iavg_radius

	sc.area, sc.perim, sc.isoparam, sc.energy, sc.avg_radius = \
		area, perim, isoparam, energy, avg_radius

	return sc


cdef inline FLOAT_T area(Site* self, FLOAT_T val) nogil:
	if isnan(<double>val):
		return self.info.site_cache.get(&self.info.site_cache,
			(self.arr_index, self.cache.iarea)
		)
	else:
		self.info.site_cache.set(&self.info.site_cache,
			(self.arr_index, self.cache.iarea), val)
		return val

cdef inline FLOAT_T perim(Site* self, FLOAT_T val) nogil:
	if isnan(<double>val):
		return self.info.site_cache.get(&self.info.site_cache,
			(self.arr_index, self.cache.iperim)
		)
	else:
		self.info.site_cache.set(&self.info.site_cache,
			(self.arr_index, self.cache.iperim), val)
		return val

cdef inline FLOAT_T isoparam(Site* self, FLOAT_T val) nogil:
	if isnan(<double>val):
		return self.info.site_cache.get(&self.info.site_cache,
			(self.arr_index, self.cache.iisoparam)
		)
	else:
		self.info.site_cache.set(&self.info.site_cache,
			(self.arr_index, self.cache.iisoparam), val)
		return val

cdef inline FLOAT_T energy(Site* self, FLOAT_T val) nogil:
	if isnan(<double>val):
		return self.info.site_cache.get(&self.info.site_cache,
			(self.arr_index, self.cache.ienergy)
		)
	else:
		self.info.site_cache.set(&self.info.site_cache,
			(self.arr_index, self.cache.ienergy), val)
		return val

cdef inline FLOAT_T avg_radius(Site* self, FLOAT_T val) nogil:
	if isnan(<double>val):
		return self.info.site_cache.get(&self.info.site_cache,
			(self.arr_index, self.cache.iavg_radius)
		)
	else:
		self.info.site_cache.set(&self.info.site_cache,
			(self.arr_index, self.cache.iavg_radius), val)
		return val


#### EdgeCacheMap Methods ####

cdef inline EdgeCacheMap init_edgecachemap(INT_T iH, INT_T ila, INT_T ida, INT_T ixij, 
				INT_T idVdv, INT_T ii2p, INT_T ila_mag, INT_T ida_mag, INT_T iphi, INT_T iB,
				INT_T iF, INT_T ilntan, INT_T icsc, INT_T size) nogil:
	cdef EdgeCacheMap ec
	ec.iH, ec.ila, ec.ida, ec.ixij, ec.idVdv, ec.ii2p, ec.ila_mag, ec.ida_mag, ec.iphi, \
		ec.iB, ec.iF, ec.ilntan, ec.icsc = iH, ila, ida, ixij, idVdv, ii2p, \
					 ila_mag, ida_mag, iphi, iB, iF, ilntan, icsc
	ec.size = size

	ec.H, ec.la, ec.da, ec.xij, ec.dVdv, ec.i2p, ec.la_mag, ec.da_mag, ec.phi, ec.B, ec.F, \
		 ec.lntan, ec.csc = H, la, da, xij, dVdv, i2p, la_mag, da_mag, phi, B, F, lntan, csc

	return ec


cdef inline Matrix2x2 H(HalfEdge* self, Matrix2x2 val) nogil:
	if isnan(<double>val.a):
		return init.Matrix2x2(
			self.info.edge_cache.get(&self.info.edge_cache,
				(self.arr_index, self.cache.iH)
			),
			self.info.edge_cache.get(&self.info.edge_cache,
				(self.arr_index, self.cache.iH+1)
			),
			self.info.edge_cache.get(&self.info.edge_cache,
				(self.arr_index, self.cache.iH+2)
			),
			self.info.edge_cache.get(&self.info.edge_cache,
				(self.arr_index, self.cache.iH+3)
			),
		)
	else:
		self.info.edge_cache.set(&self.info.edge_cache,
			(self.arr_index, self.cache.iH), val.a)
		self.info.edge_cache.set(&self.info.edge_cache,
			(self.arr_index, self.cache.iH+1), val.b)
		self.info.edge_cache.set(&self.info.edge_cache,
			(self.arr_index, self.cache.iH+2), val.c)
		self.info.edge_cache.set(&self.info.edge_cache,
			(self.arr_index, self.cache.iH+3), val.d)
		return val

cdef inline Vector2D la(HalfEdge* self, Vector2D val) nogil:
	if isnan(<double>val.x):
		return init.Vector2D(
			self.info.edge_cache.get(&self.info.edge_cache,
				(self.arr_index, self.cache.ila)
			),
			self.info.edge_cache.get(&self.info.edge_cache,
				(self.arr_index, self.cache.ila+1)
			)
		)
	else:
		self.info.edge_cache.set(&self.info.edge_cache,
			(self.arr_index, self.cache.ila), val.x)
		self.info.edge_cache.set(&self.info.edge_cache,
			(self.arr_index, self.cache.ila+1), val.y)
		return val

cdef inline Vector2D da(HalfEdge* self, Vector2D val) nogil:
	if isnan(<double>val.x):
		return init.Vector2D(
			self.info.edge_cache.get(&self.info.edge_cache,
				(self.arr_index, self.cache.ida)
			),
			self.info.edge_cache.get(&self.info.edge_cache,
				(self.arr_index, self.cache.ida+1)
			)
		)
	else:
		self.info.edge_cache.set(&self.info.edge_cache,
			(self.arr_index, self.cache.ida), val.x)
		self.info.edge_cache.set(&self.info.edge_cache,
			(self.arr_index, self.cache.ida+1), val.y)
		return val

cdef inline Vector2D xij(HalfEdge* self, Vector2D val) nogil:
	if isnan(<double>val.x):
		return init.Vector2D(
			self.info.edge_cache.get(&self.info.edge_cache,
				(self.arr_index, self.cache.ixij)
			),
			self.info.edge_cache.get(&self.info.edge_cache,
				(self.arr_index, self.cache.ixij+1)
			)
		)
	else:
		self.info.edge_cache.set(&self.info.edge_cache,
			(self.arr_index, self.cache.ixij), val.x)
		self.info.edge_cache.set(&self.info.edge_cache,
			(self.arr_index, self.cache.ixij+1), val.y)
		return val

cdef inline Vector2D dVdv(HalfEdge* self, Vector2D val) nogil:
	if isnan(<double>val.x):
		return init.Vector2D(
			self.info.edge_cache.get(&self.info.edge_cache,
				(self.arr_index, self.cache.idVdv)
			),
			self.info.edge_cache.get(&self.info.edge_cache,
				(self.arr_index, self.cache.idVdv+1)
			)
		)
	else:
		self.info.edge_cache.set(&self.info.edge_cache,
			(self.arr_index, self.cache.idVdv), val.x)
		self.info.edge_cache.set(&self.info.edge_cache,
			(self.arr_index, self.cache.idVdv+1), val.y)
		return val

cdef inline Vector2D i2p(HalfEdge* self, Vector2D val) nogil:
	if isnan(<double>val.x):
		return init.Vector2D(
			self.info.edge_cache.get(&self.info.edge_cache,
				(self.arr_index, self.cache.ii2p)
			),
			self.info.edge_cache.get(&self.info.edge_cache,
				(self.arr_index, self.cache.ii2p+1)
			)
		)
	else:
		self.info.edge_cache.set(&self.info.edge_cache,
			(self.arr_index, self.cache.ii2p), val.x)
		self.info.edge_cache.set(&self.info.edge_cache,
			(self.arr_index, self.cache.ii2p+1), val.y)
		return val

cdef inline FLOAT_T la_mag(HalfEdge* self, FLOAT_T val) nogil:
	if isnan(<double>val):
		return self.info.edge_cache.get(&self.info.edge_cache,
			(self.arr_index, self.cache.ila_mag)
		)
	else:
		self.info.edge_cache.set(&self.info.edge_cache,
			(self.arr_index, self.cache.ila_mag), val)
		return val

cdef inline FLOAT_T da_mag(HalfEdge* self, FLOAT_T val) nogil:
	if isnan(<double>val):
		return self.info.edge_cache.get(&self.info.edge_cache,
			(self.arr_index, self.cache.ida_mag)
		)
	else:
		self.info.edge_cache.set(&self.info.edge_cache,
			(self.arr_index, self.cache.ida_mag), val)
		return val

cdef inline FLOAT_T phi(HalfEdge* self, FLOAT_T val) nogil:
	if isnan(<double>val):
		return self.info.edge_cache.get(&self.info.edge_cache,
			(self.arr_index, self.cache.iphi)
		)
	else:
		self.info.edge_cache.set(&self.info.edge_cache,
			(self.arr_index, self.cache.iphi), val)
		return val

cdef inline FLOAT_T B(HalfEdge* self, FLOAT_T val) nogil:
	if isnan(<double>val):
		return self.info.edge_cache.get(&self.info.edge_cache,
			(self.arr_index, self.cache.iB)
		)
	else:
		self.info.edge_cache.set(&self.info.edge_cache,
			(self.arr_index, self.cache.iB), val)
		return val

cdef inline FLOAT_T F(HalfEdge* self, FLOAT_T val) nogil:
	if isnan(<double>val):
		return self.info.edge_cache.get(&self.info.edge_cache,
			(self.arr_index, self.cache.iF)
		)
	else:
		self.info.edge_cache.set(&self.info.edge_cache,
			(self.arr_index, self.cache.iF), val)
		return val

cdef inline FLOAT_T lntan(HalfEdge* self, FLOAT_T val) nogil:
	if isnan(<double>val):
		return self.info.edge_cache.get(&self.info.edge_cache,
			(self.arr_index, self.cache.ilntan)
		)
	else:
		self.info.edge_cache.set(&self.info.edge_cache,
			(self.arr_index, self.cache.ilntan), val)
		return val

cdef inline FLOAT_T csc(HalfEdge* self, FLOAT_T val) nogil:
	if isnan(<double>val):
		return self.info.edge_cache.get(&self.info.edge_cache,
			(self.arr_index, self.cache.icsc)
		)
	else:
		self.info.edge_cache.set(&self.info.edge_cache,
			(self.arr_index, self.cache.icsc), val)
		return val


#### VoronoiInfo Methods ####

cdef inline VoronoiInfo init_voronoiinfo(INT_T [:, ::1] sites, INT_T [:, ::1] edges,
		FLOAT_T [:, ::1] points, FLOAT_T [:, ::1] vertices, 
		FLOAT_T [:, ::1] site_cache, FLOAT_T [:, ::1] edge_cache,
		EdgeCacheMap* edge_cache_map) nogil:
	cdef VoronoiInfo info
	info.sites = init_iarray(&sites[0,0], (<INT_T>sites.shape[0], <INT_T>sites.shape[1]))
	info.edges = init_iarray(&edges[0,0], (<INT_T>edges.shape[0], <INT_T>edges.shape[1]))
	info.points = init_farray(&points[0,0], (<INT_T>points.shape[0], <INT_T>points.shape[1]))
	info.vertices = init_farray(&vertices[0,0],
		(<INT_T>vertices.shape[0], <INT_T>vertices.shape[1])
	)
	info.site_cache = init_farray(&site_cache[0,0], 
			(<INT_T>site_cache.shape[0], <INT_T>site_cache.shape[1])
	)
	info.edge_cache = init_farray(&edge_cache[0,0], 
			(<INT_T>edge_cache.shape[0], <INT_T>edge_cache.shape[1])
	)
	info.edge_cache_map = edge_cache_map

	return info


#### Site Methods ####

cdef inline Site init_site(INT_T arr_index, VoronoiInfo* info) nogil:
	cdef Site site
	site.arr_index, site.info, site.cache = arr_index, info, &SITE_CACHE_MAP
	
	site.index, site.vec, site.edge, site.edge_num = index, vec, edge, edge_num
	
	return site


cdef inline INT_T index(Site* self) nogil:
	return self.info.sites.get(&self.info.sites, (self.arr_index, 0))

cdef inline Vector2D vec(Site* self) nogil:
	return init.Vector2D(
		self.info.points.get(&self.info.points, (self.index(self), 0)), 
		self.info.points.get(&self.info.points, (self.index(self), 1))
	)

cdef inline HalfEdge edge(Site* self) nogil:
	return init.HalfEdge(
		self.info.sites.get(&self.info.sites, (self.arr_index, 1)), self.info
	)

cdef inline INT_T edge_num(Site* self) nogil:
	return self.info.sites.get(&self.info.sites, (self.arr_index, 2))


#### HalfEdge Methods ####

cdef inline HalfEdge init_halfedge(INT_T arr_index, VoronoiInfo* info) nogil:
	cdef HalfEdge edge
	edge.arr_index, edge.info, edge.cache = arr_index, info, info.edge_cache_map
	edge.orig_arr_index = arr_index

	edge.origin_index, edge.origin, edge.face, edge.next, edge.prev, edge.twin, edge.get_H = \
		origin_index, origin, face, edge_next, prev, twin, get_H
	
	return edge


cdef inline INT_T origin_index(HalfEdge* self) nogil:
	return self.info.edges.get(&self.info.edges, (self.arr_index, 0))

cdef inline Vector2D origin(HalfEdge* self) nogil:
	return init.Vector2D(
		self.info.vertices.get(&self.info.vertices, (self.origin_index(self), 0)),
		self.info.vertices.get(&self.info.vertices, (self.origin_index(self), 1))
	)

cdef inline Site face(HalfEdge* self) nogil:
	return init.Site(
		self.info.edges.get(&self.info.edges, (self.arr_index, 1)), self.info
	)

cdef inline HalfEdge edge_next(HalfEdge* self) nogil:

	return init.HalfEdge(
		self.info.edges.get(&self.info.edges, (self.arr_index, 2)), self.info
	)

cdef inline HalfEdge prev(HalfEdge* self) nogil:
	return init.HalfEdge(
		self.info.edges.get(&self.info.edges, (self.arr_index, 3)), self.info
	)

cdef inline HalfEdge twin(HalfEdge* self) nogil:
	return init.HalfEdge(
		self.info.edges.get(&self.info.edges, (self.arr_index, 4)), self.info
	)

cdef inline Matrix2x2 get_H(HalfEdge* self, Site xi) nogil:
	cdef INT_T this_e = self.origin_index(self)
	cdef HalfEdge s_e = xi.edge(&xi)
	cdef INT_T i

	for i in range(xi.edge_num(&xi)):
		if s_e.origin_index(&s_e) == this_e:
			return s_e.cache.H(&s_e, NAN_MATRIX)
		s_e = s_e.next(&s_e)
	return init.Matrix2x2(0.0, 0.0, 0.0, 0.0)


cdef class VoronoiContainer:
	"""
	Class for Voronoi diagrams, stored in a modified DCEL.
	:param n: [int] how many sites to generate.
	:param w: [float] width of the bounding domain.
	:param h: [float] height of the bounding domain.
	:param r: [float] radius of zero energy circle.
	:param sites: np.ndarray collection of sites.
	"""

	def __init__(VoronoiContainer self, INT_T n, FLOAT_T w, FLOAT_T h, FLOAT_T r, object site_arr):
		self.n, self.w, self.h, self.r = n, w, h, r
		self.dim = [w, h]

		self.calculate_voronoi(site_arr.astype(FLOAT))
		self.generate_dcel()	

		self.common_cache()
		self.precompute()
		self.calc_grad()
		self.get_statistics()


	cdef void calculate_voronoi(VoronoiContainer self, 
								np.ndarray[FLOAT_T, ndim=2] site_arr) except *:
		"""
		Does all necessary computation and caching once points are set.
		:param site_arr: initial points for this container.
		""" 
		global SYMM
		cdef np.ndarray[FLOAT_T, ndim=2] symm = np.asarray(SYMM).reshape(9,2)
		cdef np.ndarray[FLOAT_T, ndim=1] dim = np.asarray(self.dim)
		cdef np.ndarray[FLOAT_T, ndim=2] full_site_arr = np.empty((self.n*9+8, 2), dtype=FLOAT)
		
		# Generate periodic sites and sites that bound periodic sites.
		cdef INT_T i
		for i in range(9):
			full_site_arr[self.n*i:self.n*(i+1)] = site_arr + symm[i]*dim
			if i > 0:
				full_site_arr[9*self.n+i-1] = dim/2 + 2*dim*symm[i]

		# Use SciPy to compute the Voronoi set.
		self.scipy_vor = scipy.spatial.Voronoi(full_site_arr)
		self.points = self.scipy_vor.points
		self.vertices = self.scipy_vor.vertices


	cdef void generate_dcel(VoronoiContainer self) except *:
		cdef INT_T npoints = self.n*9+8
		cdef array.array int_tmplt = array.array('q', []) 
		
		cdef np.ndarray[INT_T, ndim=1] offsets = np.zeros(self.n*9+1, dtype=INT)
		cdef array.array vert_indices = array.clone(int_tmplt, 0, False)
	
		# Flatten regions into array, so it can be used later.
		cdef INT_T i
		for i in range(self.n*9):
			verts = self.scipy_vor.regions[self.scipy_vor.point_region[i]]	
			offsets[i+1] = offsets[i] + len(verts) # Build offsets.
			vert_indices.extend(array.array('q', verts))	# Flatten

		# Get vertices of original N sites.
		cdef np.ndarray[INT_T, ndim=1] vert_indices_np = np.asarray(vert_indices)
		cdef np.ndarray[INT_T, ndim=1] border_sites = np.unique(np.searchsorted(
			np.asarray(offsets),	# Check indices where below matches would be inserted
			np.nonzero(np.isin(		# Indices of other verts being in bound verts.
				vert_indices_np[offsets[self.n]:], 	# Rest of the verts to check.
				np.unique(vert_indices_np[:offsets[self.n]])	# Bound verts
			))[0] + offsets[self.n], 
			side='right'	# If on index == offset_number, should be part of the next site.
		) - 1)	# Subtract by one to get actual site number.	

		cdef INT_T border_num = len(border_sites)

		# Build sites array.
		# [Site Index, Edge Index/Offset, Edge Count]
		self.sites = np.empty((self.n+border_num, 3), dtype=INT)
		self.sites.base[:self.n, 0] = np.arange(self.n, dtype=INT)
		self.sites.base[self.n:, 0] = border_sites
		self.sites.base[:self.n+1, 1] = offsets[:self.n+1] 
		for i in range(self.n):
			self.sites[i, 2] = self.sites[i+1, 1] - self.sites[i, 1]

		cdef INT_T edge_count = offsets[self.n]
		cdef INT_T diff
		for i in range(border_num):
			diff = offsets[border_sites[i]+1] - offsets[border_sites[i]]
			edge_count += diff
			self.sites[self.n+i, 2] = diff 
			if i < border_num - 1:
				self.sites[self.n+i+1, 1] = self.sites[self.n+i, 1] + diff
		
		# Build edges array
		# [Origin Index, Site Index, Next Index, Prev Index, Twin Index]
		self.edges = np.empty((edge_count, 5), dtype=INT)
		cdef np.ndarray[INT_T, ndim=1] site_verts
		cdef INT_T j, site_i, edge_i, edge_offset, vert_num, twin_index, prev_res

		edge_indices = dict()

		for i in range(self.n + border_num):
			site_i = self.sites[i, 0]
			edge_offset = self.sites[i, 1]
			site_verts = vert_indices_np[offsets[site_i]:offsets[site_i+1]]

			# Scipy outputs sorted vertices, but reverse if not counterclockwise.
			if not VoronoiContainer.sign(self.points[site_i], 
					self.vertices[site_verts[0]], self.vertices[site_verts[1]]):
				site_verts = np.flip(site_verts)

			vert_num = offsets[site_i+1] - offsets[site_i] 

			for j in range(vert_num):
				edge_i = edge_offset+j
				self.edges[edge_i, 0] = site_verts[j]
				self.edges[edge_i, 1] = i
				# Add vert_num because of C modulo to get always positive.
				self.edges[edge_i, 2] = (j+vert_num+1) % vert_num + edge_offset
				self.edges[edge_i, 3] = (j+vert_num-1) % vert_num + edge_offset

				# Get reversed tuple to theck for twin.
				twin_index = edge_indices.get(
					(site_verts[(j+1) % vert_num], site_verts[j]
				), -1)

				self.edges[edge_i, 4] = twin_index
				if twin_index == -1:
					edge_indices[(site_verts[j], site_verts[(j+1) % vert_num])] = \
						j + edge_offset
				else:
					self.edges[twin_index, 4] = j + edge_offset

		self.site_cache = np.empty((self.n + border_num, 5), dtype=FLOAT)
		self.edge_cache = np.empty((edge_count, self.edge_cache_map.size), dtype=FLOAT)


	cdef void common_cache(VoronoiContainer self) except *:
		cdef VoronoiInfo info = init.VoronoiInfo(self.sites, self.edges, self.points, 
			self.vertices, self.site_cache, self.edge_cache, self.edge_cache_map)

		cdef Site xi
		cdef HalfEdge em, ep
		cdef Vector2D p, q, la, da, Rla

		cdef FLOAT_T [:] area = np.zeros(self.sites.shape[0], dtype=FLOAT)
		cdef FLOAT_T [:] perim = np.zeros(self.sites.shape[0], dtype=FLOAT)

		cdef INT_T i, j
		cdef FLOAT_T e_area, la_mag
		for i in prange(self.sites.shape[0], nogil=True):
			xi = init.Site(i, &info)
			em = xi.edge(&xi)
			for j in range(xi.edge_num(&xi)):
				ep = em.next(&em)
				p, q = em.origin(&em), ep.origin(&ep)
				la, da = q.copy.vsub(&q, p), p.copy.vsub(&p, xi.vec(&xi))	# vp - vm, vm - xi

				la_mag = la.mag(&la)
				e_area = la.dot(&la, da.rot(&da))
				Rla = la.rot(&la)

				em.cache.la(&em, la)
				em.cache.la_mag(&em, la_mag)
				em.cache.da(&em, da)
				em.cache.da_mag(&em, da.mag(&da))
				em.cache.xij(&em, Rla.copy.smul(&Rla, -e_area/la.dot(&la, la)))

				if info.edge_cache_map.iF != -1:
					em.cache.F(&em, e_area)

				area[i] += e_area
				perim[i] += la_mag

				em = em.next(&em)

			xi.cache.area(&xi, area[i]/2)
			xi.cache.perim(&xi, perim[i])
			xi.cache.isoparam(&xi, 2*PI*area[i]/(perim[i]*perim[i]))


	@staticmethod
	cdef inline Matrix2x2 calc_H(HalfEdge em, HalfEdge ep) nogil:
		cdef Vector2D xmv, xpv, im, mp, right, Rpm, Rim, f
		cdef Matrix2x2 h
		cdef FLOAT_T im2, mp2

		# Vectors from xi to xm and xp.
		xmv, xpv = em.cache.xij(&em, NAN_VECTOR), ep.cache.xij(&ep, NAN_VECTOR)
		im, mp = xmv.copy.neg(&xmv), xmv.copy.vsub(&xmv, xpv)	# -xmv, xmv - xpv
		im2, mp2 = -(xmv.dot(&xmv, xmv)), xmv.dot(&xmv, xmv) - xpv.dot(&xpv, xpv)
		# (-xmv*xmv, xmv*xmv - xpv*xpv)
		right = init.Vector2D(im2, mp2)
		Rpm, Rim = R.vecmul(&R, mp.copy.neg(&mp)), im.rot(&im)	# R*-mp, R*im
		
		h = init.Matrix2x2(Rpm.x, Rim.x, Rpm.y, Rim.y)	# [Rpm | Rim], h is temporary.
		f = h.vecmul(&h, right)	# [Rpm | Rim]*right
		h = R.copy.smul(&R, mp2*(2*mp.dot(&mp, Rim)))	# fp*g, g is a scalar.
		# (fp*g - f*gp)/(g**2). f is a column vector, gp = 2*Rpm is a row vector.
		h.self.msub(&h, init.Matrix2x2(	
			f.x*2*Rpm.x, f.x*2*Rpm.y, f.y*2*Rpm.x, f.y*2*Rpm.y
		))
		h.self.sdiv(&h, (2*mp.dot(&mp, Rim))**2)

		return h


	@staticmethod
	cdef inline bint sign(FLOAT_T [::1] ref, FLOAT_T [::1] p, FLOAT_T [::1] q):
		"""
		Outputs if p2 - self is counterclockwise of p1 - self.
		:param p1: [List[float]] first vector
		:param p2: [List[float]] second vector
		:return: [bool] returns if counterclockwise.
		"""
		return ((q[0] - ref[0])*-(p[1] - ref[1]) + \
				  (q[1] - ref[1])*(p[0] - ref[0])) >= 0 

		# global ROT
		# cdef np.ndarray[FLOAT_T, ndim=2] rot = np.asarray(ROT).reshape(2,2)
		# return (q - ref).dot(rot.dot(p - ref)) >= 0

	cdef void precompute(self) except *:
		pass

	cdef void calc_grad(self) except *:
		pass

	cdef void get_statistics(self) except *:
		self.stats = {}
		cache = self.site_cache[:self.n, :]

		self.stats["site_areas"] = np.asarray(cache[:, SITE_CACHE_MAP.iarea])
		self.stats["site_edge_count"] = np.asarray(self.sites[:self.n, 2])

		self.stats["site_isos"] = np.asarray(cache[:, SITE_CACHE_MAP.iisoparam])
		self.stats["site_energies"] = np.asarray(cache[:, SITE_CACHE_MAP.ienergy])
		self.stats["avg_radius"] = np.asarray(cache[:, SITE_CACHE_MAP.iavg_radius])
		
		self.stats["isoparam_avg"] = self.stats["site_areas"] / \
						(PI*self.stats["avg_radius"]**2)
		
		edges = np.asarray(self.edges)

		mask = np.nonzero(edges[:, 0] != -1)[0]
		all_edges = mask[(mask % 2 == 0)] 
		caches = edges[all_edges, 4]

		edge_cache = np.asarray(self.edge_cache)

		self.stats["edge_lengths"] = edge_cache[caches, self.edge_cache_map.ila_mag]

	@property
	def site_arr(self):
		return np.asarray(self.points[:self.n], dtype=FLOAT)

	@property
	def vor_data(self):
		return self.scipy_vor

	@property
	def gradient(self):
		return np.asarray(self.grad, dtype=FLOAT)
	
	def add_sites(self, add):
		return (self.site_arr + add) % np.asarray(self.dim, dtype=FLOAT)
	
	def iterate(self, FLOAT_T step):
		k1 = self.gradient
		k2 = self.__class__(self.n, self.w, self.h, self.r,
				self.add_sites(step*k1)
		).gradient

		return (step/2)*(k1+k2), k1


	def hessian(self, d: float) -> np.ndarray:
		"""
		Obtains the approximate Hessian.
		:param d: [float] small d for approximation.
		:return: 2Nx2N array that represents Hessian.
		"""
		HE = np.zeros((2*self.n, 2*self.n))
		new_sites = np.copy(self.site_arr)	# Maintain one copy for speed.
		for i in range(self.n):
			for j in range(2):
				mod = self.w if j == 0 else self.h
				new_sites[i][j] = (new_sites[i][j] + d) % mod 
				Ep = self.__class__(self.n, self.w, self.h, self.r, new_sites)
				new_sites[i][j] = (new_sites[i][j] - 2*d) % mod
				Em = self.__class__(self.n, self.w, self.h, self.r, new_sites)
				new_sites[i][j] = (new_sites[i][j] + d) % mod

				HE[:, 2*i+j] = ((Ep.gradient - Em.gradient)/(2*d)).flatten()

		# Average out discrepencies, since it should be symmetric.
		for i in range(2*self.n):
			for j in range(i, 2*self.n):
				HE[i][j] = (HE[i][j] + HE[j][i])/2
				HE[j][i] = HE[i][j]

		return HE

