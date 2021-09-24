cdef class AreaEnergy(VoronoiContainer):
	"""
	Class for formulas relevant to the Area energy.
	:param n: [int] how many sites to generate.
	:param w: [float] width of the bounding domain.
	:param h: [float] height of the bounding domain.
	:param r: [float] radius of zero energy circle.
	:param sites: [np.ndarray] collection of sites.
	"""

	attr_str = "area"
	title_str = "Area"

	def __init__(AreaEnergy self, INT_T n, FLOAT_T w, FLOAT_T h, FLOAT_T r,
					np.ndarray[FLOAT_T, ndim=2] site_arr):
		self.edge_cache_map = &AREA_EDGE_CACHE_MAP
		self.energy = 0.0

		super().__init__(n, w, h, r, site_arr)
		self.minimum = (<FLOAT_T>n)*(w*h/(<FLOAT_T>n)-PI*r**2)**2


	cdef void precompute(self) except *:
		cdef VoronoiInfo info = init.VoronoiInfo(self.sites, self.edges, self.points, 
			self.vertices, self.site_cache, self.edge_cache, self.edge_cache_map)

		cdef Site xi
		cdef HalfEdge em, e, ep
		cdef Vector2D vdiff
		cdef FLOAT_T A = PI*self.r**2
		cdef FLOAT_T energy = 0

		cdef INT_T i, j
		for i in prange(self.sites.shape[0], nogil=True):
			xi = init.Site(i, &info)
			e = xi.edge(&xi)
			xi.cache.energy(&xi, 
				(xi.cache.area(&xi, NAN) - A)**2
			)
			if i < self.n:
				energy += xi.cache.energy(&xi, NAN)
			
			for j in range(xi.edge_num(&xi)):
				em, ep = e.prev(&e), e.next(&e)
				vdiff = em.origin(&em)
				vdiff.self.vsub(&vdiff, ep.origin(&ep))
				e.cache.dVdv(&e, R.vecmul(&R, vdiff))
				e.cache.H(&e, VoronoiContainer.calc_H(em, e))

				e = e.next(&e)

		self.energy = energy


	cdef void calc_grad(self) except *:
		cdef VoronoiInfo info = init.VoronoiInfo(self.sites, self.edges, self.points, 
				self.vertices, self.site_cache, self.edge_cache, self.edge_cache_map)

		cdef Site xi, xf
		cdef HalfEdge e, f
		cdef Vector2D dedxi_p
		cdef BitSet edge_set

		cdef INT_T num_edges = self.edges.shape[0]
		cdef FLOAT_T A = PI*self.r**2

		cdef FLOAT_T [:, ::1] dedx = np.zeros((self.n, 2), dtype=FLOAT)

		cdef INT_T i, j
		for i in prange(self.n, nogil=True):
			xi = init.Site(i, &info)
			e = xi.edge(&xi)
			edge_set = init.BitSet(num_edges)
			for j in range(xi.edge_num(&xi)):	# Looping through site edges.
				f = e
				while True:	# Circling this vertex.
					if not edge_set.add(&edge_set, f.arr_index):
						xf = f.face(&f)
						dedxi_p = f.cache.dVdv(&f, NAN_VECTOR)	#dVdv
						dedxi_p.self.smul(&dedxi_p, xf.cache.area(&xf, NAN) - A)
						dedxi_p.self.matmul(&dedxi_p, e.cache.H(&e, NAN_MATRIX))
						dedx[i][0] -= dedxi_p.x
						dedx[i][1] -= dedxi_p.y

					f = f.twin(&f)
					f = f.next(&f)
					if f.arr_index == e.arr_index:
						break

				e = e.next(&e)
			edge_set.free(&edge_set)
		self.grad = dedx


cdef class RadialALEnergy(VoronoiContainer):
	"""
	Class for formulas relevant to the Area energy.
	:param n: [int] how many sites to generate.
	:param w: [float] width of the bounding domain.
	:param h: [float] height of the bounding domain.
	:param r: [float] radius of zero energy circle.
	:param sites: [np.ndarray] collection of sites.
	"""

	attr_str = "radial-al"
	title_str = "Radial[AL]"


	def __init__(AreaEnergy self, INT_T n, FLOAT_T w, FLOAT_T h, FLOAT_T r,
					np.ndarray[FLOAT_T, ndim=2] site_arr):
		#self.edge_cache_map = &AREA_EDGE_CACHE_MAP
		self.energy = 0.0

		super().__init__(n, w, h, r, site_arr)


	cdef void precompute(self) except *:
		cdef VoronoiInfo info = init.VoronoiInfo(self.sites, self.edges, self.points, 
			self.vertices, self.site_cache, self.edge_cache, self.edge_cache_map)

		pass


	cdef void calc_grad(self) except *:
		cdef VoronoiInfo info = init.VoronoiInfo(self.sites, self.edges, self.points, 
				self.vertices, self.site_cache, self.edge_cache, self.edge_cache_map)

		pass


cdef class RadialTEnergy(VoronoiContainer):
	"""
	Class for formulas relevant to the Area energy.
	:param n: [int] how many sites to generate.
	:param w: [float] width of the bounding domain.
	:param h: [float] height of the bounding domain.
	:param r: [float] radius of zero energy circle.
	:param sites: [np.ndarray] collection of sites.
	"""

	attr_str = "radial-t"
	title_str = "Radial[T]"
	def __init__(AreaEnergy self, INT_T n, FLOAT_T w, FLOAT_T h, FLOAT_T r,
					np.ndarray[FLOAT_T, ndim=2] site_arr):
		self.edge_cache_map = &RADIALT_EDGE_CACHE_MAP
		self.energy = 0.0

		super().__init__(n, w, h, r, site_arr)


	cdef void precompute(self) except *:
		cdef VoronoiInfo info = init.VoronoiInfo(self.sites, self.edges, self.points, 
			self.vertices, self.site_cache, self.edge_cache, self.edge_cache_map)

		cdef Site xi
		cdef HalfEdge em, e
		cdef Vector2D Rnla

		# All energy has a 2pir_0 term.
		cdef FLOAT_T [:] site_energy = np.full(self.sites.shape[0], TAU*self.r**2)
		cdef FLOAT_T [:] avg_radii = np.zeros(self.sites.shape[0])
		cdef FLOAT_T energy, r0, t, tp, B, lntan, csc
		energy, r0 = 0, self.r

		cdef INT_T i, j
		for i in prange(self.sites.shape[0], nogil=True):
			xi = init.Site(i, &info)
			e = xi.edge(&xi)
			for j in range(xi.edge_num(&xi)):
				em = e.prev(&e)
				e.cache.H(&e, VoronoiContainer.calc_H(em, e))
				t = Calc.phi(e)

				e.cache.phi(&e, t)
				Rnla = e.cache.la(&e, NAN_VECTOR)
				Rnla.self.neg(&Rnla)
				Rnla = Rnla.rot(&Rnla)

				if Rnla.x < 0:
					e.cache.B(&e, -<FLOAT_T>acos(<double>(Rnla.y/e.cache.la_mag(&e, NAN))))
				else:
					e.cache.B(&e, <FLOAT_T>acos(<double>(Rnla.y/e.cache.la_mag(&e, NAN))))		
				
				e.cache.i2p(&e, Calc.I2(e, r0, t))
				e = e.next(&e)

			# For looping again to calculate integrals.
			em = xi.edge(&xi)
			for j in range(xi.edge_num(&xi)):
				e = em.next(&em)
				B = em.cache.B(&em, NAN)
				t, tp = em.cache.phi(&em, NAN), e.cache.phi(&e, NAN)

				lntan = <FLOAT_T>(log(fabs(tan(<double>((tp+B)/2))))) - \
							<FLOAT_T>(log(fabs(tan(<double>((t+B)/2)))))

				csc = 1/(<FLOAT_T>(sin(<double>(tp+B)))) - \
						1/(<FLOAT_T>(sin(<double>(t+B))))
				
				em.cache.lntan(&em, lntan)
				em.cache.csc(&em, csc)

				avg_radii[i] += (em.cache.F(&em, NAN)/em.cache.la_mag(&em, NAN))*lntan

				em = em.next(&em)

			site_energy[i] += 2*(xi.cache.area(&xi, NAN) - r0*avg_radii[i])

			xi.cache.avg_radius(&xi, avg_radii[i]/TAU)
			xi.cache.energy(&xi, site_energy[i])
			if i < self.n:
				energy += site_energy[i]

		self.energy = energy


	cdef void calc_grad(self) except *:
		cdef VoronoiInfo info = init.VoronoiInfo(self.sites, self.edges, self.points, 
				self.vertices, self.site_cache, self.edge_cache, self.edge_cache_map)

		cdef Site xi
		cdef HalfEdge e, fm, f
		cdef Vector2D dedxi_p
		cdef BitSet edge_set
		
		cdef INT_T num_edges = self.edges.shape[0]
		cdef FLOAT_T r0 = self.r

		cdef FLOAT_T [:, ::1] dedx = np.zeros((self.n, 2), dtype=FLOAT)

		cdef INT_T i, j
		for i in prange(self.n, nogil=True):
			xi = init.Site(i, &info)
			e = xi.edge(&xi)
			edge_set = init.BitSet(num_edges)

			for j in range(xi.edge_num(&xi)):	# Looping through site edges.
				f = e
				while True:	# Circling this vertex.
					fm = f.prev(&f)
					if not edge_set.add(&edge_set, f.arr_index):
						dedxi_p = Calc.radialt_edge_grad(f, xi, r0)
						dedx[i][0] -= dedxi_p.x
						dedx[i][1] -= dedxi_p.y

					if not edge_set.add(&edge_set, fm.arr_index):
						dedxi_p = Calc.radialt_edge_grad(fm, xi, r0)
						dedx[i][0] -= dedxi_p.x
						dedx[i][1] -= dedxi_p.y

						
					f = f.twin(&f)
					f = f.next(&f)
					
					if f.arr_index == e.arr_index:
						break

				e = e.next(&e)
			edge_set.free(&edge_set)
		self.grad = dedx


cdef class Calc:
	@staticmethod
	cdef inline FLOAT_T phi(HalfEdge e) nogil:
		cdef Vector2D da = e.cache.da(&e, NAN_VECTOR)
		cdef FLOAT_T angle = <FLOAT_T>acos(<double>(da.x/e.cache.da_mag(&e, NAN)))
		return angle if da.y >= 0 else TAU - angle


	@staticmethod
	cdef inline Vector2D I2(HalfEdge e, FLOAT_T r0, FLOAT_T t) nogil:
		cdef Vector2D Rda = e.cache.da(&e, NAN_VECTOR)
		Rda = Rda.rot(&Rda)
		Rda.self.sdiv(&Rda, e.cache.da_mag(&e, NAN))

		return Rda


	@staticmethod 
	cdef Vector2D radialt_edge_grad(HalfEdge e, Site xi, FLOAT_T r0) nogil:
		cdef Site xe
		cdef HalfEdge ep
		cdef Vector2D Rda, i2ps, fp, gterms, q
		cdef Matrix2x2 ha, hap, hdiff

		cdef FLOAT_T t1, t2, lntan, csc, sinB, cosB, sinBp, cosBp, F, A, B

		xe = e.face(&e)
		ep = e.next(&e)
		F, A, B = e.cache.F(&e, NAN), e.cache.la_mag(&e, NAN), e.cache.B(&e, NAN)
		t1, t2 = e.cache.phi(&e, NAN), ep.cache.phi(&ep, NAN)

		lntan, csc = e.cache.lntan(&e, NAN), e.cache.csc(&e, NAN)

		sinB, cosB = <FLOAT_T>(sin(<double>(B))), <FLOAT_T>(cos(<double>(B)))
		sinBp, cosBp = <FLOAT_T>(sin(<double>(B-PI_2))), \
						<FLOAT_T>(cos(<double>(B-PI_2)))


		ha, hap = e.get_H(&e, xi), ep.get_H(&ep, xi)
		hdiff = hap.copy.msub(&hap, ha)
		# If edge is part of differentiated site.
		if xe.index(&xe) == xi.index(&xi):
			ha.self.msub(&ha, init.Matrix2x2(1.0, 0.0, 0.0, 1.0))
			hap.self.msub(&hap, init.Matrix2x2(1.0, 0.0, 0.0, 1.0))

		i2ps = ep.cache.i2p(&ep, NAN_VECTOR)
		i2ps.self.matmul(&i2ps, hap)

		q = e.cache.i2p(&e, NAN_VECTOR)
		q.self.matmul(&q, ha)

		i2ps.self.vsub(&i2ps, q)

		Rda = e.cache.da(&e, NAN_VECTOR)
		Rda = Rda.rot(&Rda)

		fp = e.cache.la(&e, NAN_VECTOR)
		fp.self.matmul(&fp, R.copy.matmul(&R, ha))
		fp.self.vadd(&fp, Rda.copy.matmul(&Rda, hdiff))
		fp.self.smul(&fp, lntan/A)
		
		gterms = init.Vector2D(
			cosBp*lntan + sinBp*csc,
			cosB*lntan + sinB*csc
		)
		gterms.self.smul(&gterms, -F/A**2)

		gterms = gterms.rot(&gterms)
		gterms.self.matmul(&gterms, hdiff)

		fp.self.vadd(&fp, gterms)

		i2ps.self.vadd(&i2ps, fp)
		i2ps.self.smul(&i2ps, -2*r0)

		return i2ps