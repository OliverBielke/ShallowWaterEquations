import numpy as np
from simulation import Simulation
from Animation import Animation

class Run:
	def __init__(self, type:str, max_time:float, integration:str="RK4"):
		self.type = type
		self.max_time = max_time
		self.integration = integration
	

	def run_simulation(self, x_steps:int, order_of_accuracy:int=2) -> None:
		"""Simulate and animate the simulation. """

		x_list, b, time_step, L_l, L_r, q_0, L = self.__get_type_constants(x_steps=x_steps)
		

		s = Simulation(max_time=self.max_time, bottom_list=b, time_step=time_step, \
					L_l=L_l, q_0=q_0, L_r=L_r, L=L, integration=self.integration, \
						order_of_accuracy=order_of_accuracy, type=self.type)
		s.simulate()

		

		b_matrix = [b for _ in s.t_list]

		a = Animation(b_matrix=b_matrix, h_matrix=s.h_list, t_list=s.t_list, x_list=x_list)
		a.animate()


	def disp_last_image(self, x_steps:int, order_of_accuracy:int=2) -> None:
		"""Runs the simulation and shows the last time step. """
		x_list, b, time_step, L_l, L_r, q_0, L = self.__get_type_constants(x_steps=x_steps)
		

		s = Simulation(max_time=self.max_time, bottom_list=b, time_step=time_step, \
					L_l=L_l, q_0=q_0, L_r=L_r, L=L, integration=self.integration, \
						order_of_accuracy=order_of_accuracy, type=self.type)
		s.simulate()

		b_matrix = [b for _ in s.t_list]

		a = Animation(b_matrix=b_matrix, h_matrix=s.h_list, t_list=s.t_list, x_list=x_list)

		a.plot_last()


	def error_calc(self): #TODO
		"""Prints the errors for the given type of simulation. """

		if self.type == "lake at rest":
			
			for m in [50, 100, 200, 400, 800]:
				print("m =", m)
				for OOA in [2, 3, 5]:
					print("Order of accuracy =", OOA)

					_, b, time_step, L_l, L_r, q_0, L = self.__get_type_constants(x_steps=m)

					s = Simulation(max_time=self.max_time, bottom_list=b, time_step=time_step, \
								L_l=L_l, q_0=q_0, L_r=L_r, L=L, integration=self.integration, \
									order_of_accuracy=OOA, type=self.type)
					
					s.simulate()

					qandb = s.q + s.c.b_overline

					error = self.__logL_2_error(qandb=qandb)

					print("log_10 L^2 =", error)
					print("")

	
	def __logL_2_error(self, qandb:np.ndarray) -> float:
		"""Returns the error of given the type. """
		
		if self.type == "lake at rest":
			h_compare = np.full(shape=(len(qandb)//2, 1), fill_value=0.5)
			u_compare = np.zeros(shape=(len(qandb)//2, 1))
			q_compare = np.append(h_compare, u_compare)

			square_error = 0
			for i in range(len(qandb)):
				square_error += (qandb[i] - q_compare[i])**2
			
			return np.log10(np.sqrt(square_error)/len(qandb))
		

	def flow_error_calc(self):
		"""Prints the errors for the given type of simulation. """

		if self.type == "flowing lake":
		
			for m in [51, 101, 201, 401, 801]:
				print("m =", m)
				for OOA in [2, 3, 5]:
					print("Order of accuracy =", OOA)

					x_list, b, time_step, L_l, L_r, q_0, L = self.__get_type_constants(x_steps=m)

					s = Simulation(max_time=self.max_time, bottom_list=b, time_step=time_step, \
								L_l=L_l, q_0=q_0, L_r=L_r, L=L, integration=self.integration, \
									order_of_accuracy=OOA, type=self.type)
					
					s.simulate()

					h = s.q[:m]
					error = self.__flow_logL_2_error(h=h[:(m-1)//10+1], b_list=b[:(m-1)//10+1])

					print("log_10 L^2 =", error)
					print("")

	def __flow_logL_2_error(self, h:np.ndarray, b_list:np.ndarray) -> float:
		"""Returns the error of given the type. """
		
		m = len(h)

		exact_list = []
		for i in range(m):
			a = 1
			b = b_list[i] - 4.42**2/(2*9.81*h[-1]**2) - h[-1]
			d = 4.42**2/(2*9.81)
			exact_list.append(self.__cubic_solve(a=a, b=b, d=d) + b_list[i])

		square_error = 0
		for i in range(m):
			square_error += (h[i] - exact_list[i])**2
		
		
		return np.log10(np.sqrt(square_error)/m)


	def __cubic_solve(self, a:float, b:float, d:float):
		xi = (-1+np.emath.sqrt(-3))/2
		
		delta0 = b**2
		delta1 = 2*b**3 + 27*a**2*d

		inin =delta1**2 - 4*delta0**3
		C_in = (delta1 + np.emath.sqrt(inin))
		C = (C_in/2)**(1/3)

		res = -(1/3/a)*(b + C + delta0/C)
		res2 = -(1/3/a)*(b + xi*C + delta0/xi/C)
		res3 = -(1/3/a)*(b + xi**2*C + delta0/xi**2/C)

		return res2


	def gaus_error_calc(self): #TODO
		"""Prints the errors for the given type of simulation. """

		if self.type == "gaussian pulse":
			
			_, b, time_step, L_l, L_r, q_0, L = self.__get_type_constants(x_steps=1601)

			s = Simulation(max_time=self.max_time, bottom_list=b, time_step=time_step, \
								L_l=L_l, q_0=q_0, L_r=L_r, L=L, integration=self.integration, \
									order_of_accuracy=5, type=self.type)
			
			s.simulate()

			compare = s.q

			#compare = np.zeros(3202)
		
			for m in [51, 101, 201, 401, 801]:
				print("m =", m)
				for OOA in [2, 3, 5]:
					print("Order of accuracy =", OOA)

					_, b, time_step, L_l, L_r, q_0, L = self.__get_type_constants(x_steps=m)

					s = Simulation(max_time=self.max_time, bottom_list=b, time_step=time_step, \
								L_l=L_l, q_0=q_0, L_r=L_r, L=L, integration=self.integration, \
									order_of_accuracy=OOA, type=self.type)
					
					s.simulate()

					qandb = s.q

					error = self.__gaus_logL_2_error(qandb=qandb, q_compare=compare)

					print("log_10 L^2 =", error)
					print("")

	
	def __gaus_logL_2_error(self, qandb:np.ndarray, q_compare:np.ndarray) -> float:
		"""Returns the error of given the type. """
		
		m = len(qandb) // 2

		diff = (len(q_compare)//2-1) // (m-1)


		square_error = 0
		for i in range(len(qandb)//2):
			square_error += (qandb[i] - q_compare[i*diff])**2

		for j in range(len(qandb)//2, len(qandb)):
			square_error += (qandb[j] - q_compare[j*(diff)-(diff-1)])**2
		
		
		return np.log10(np.sqrt(square_error)/len(qandb))


	def convergence_calc(self):
		"""Prints the errors for the given type of simulation. """

		if self.type == "gaussian pulse":
			
			_, b, time_step, L_l, L_r, q_0, L = self.__get_type_constants(x_steps=1601)

			s = Simulation(max_time=self.max_time, bottom_list=b, time_step=time_step, \
								L_l=L_l, q_0=q_0, L_r=L_r, L=L, integration=self.integration, \
									order_of_accuracy=5, type=self.type)
			
			s.simulate()

			compare = s.q
		

			for OOA in [2, 3, 5]:
				print("Order of accuracy =", OOA)
				_, b, time_step, L_l, L_r, q_0, L = self.__get_type_constants(x_steps=51)

				s = Simulation(max_time=self.max_time, bottom_list=b, time_step=time_step, \
							L_l=L_l, q_0=q_0, L_r=L_r, L=L, integration=self.integration, \
								order_of_accuracy=OOA, type=self.type)
				
				s.simulate()

				qandb = s.q

				error2 = self.__H_error(qandb=qandb, q_compare=compare, s=s)

				for m in [101, 201, 401, 801]:
					print("m =", m)
					
					error1 = error2

					_, b, time_step, L_l, L_r, q_0, L = self.__get_type_constants(x_steps=m)

					s = Simulation(max_time=self.max_time, bottom_list=b, time_step=time_step, \
								L_l=L_l, q_0=q_0, L_r=L_r, L=L, integration=self.integration, \
									order_of_accuracy=OOA, type=self.type)
					
					s.simulate()

					qandb = s.q

					error2 = self.__H_error(qandb=qandb, q_compare=compare, s=s)

					print("c =", np.log10(error1/error2)/np.log10(m/((m-1)/2+1)))
					print("")


	def __H_error(self, qandb:np.ndarray, q_compare:np.ndarray, s:Simulation) -> float:
		"""Returns the error of given the type. """
		
		m = len(qandb) // 2

		diff = (len(q_compare)//2-1) // (m-1)


		error_array = np.zeros(len(qandb))
		for i in range(len(qandb)//2):
			error_array[i] = (qandb[i] - q_compare[i*diff])

		for j in range(len(qandb)//2, len(qandb)):
			error_array[j] += (qandb[j] - q_compare[j*(diff)-(diff-1)])
		
		
		return np.sqrt(np.dot(error_array, np.dot(np.kron(s.c.I_2, s.c.H), error_array)))


	def __get_type_constants(self, x_steps:int) -> tuple:
		"""Get the constants for the given type of simulation. \
			Returns x_list, b, time_step, L_l, L_r, q_0. """

		if self.type == "dam break":
			L = 1
			x_list = [x*L/(x_steps-1) for x in range(x_steps)]
			b = np.zeros(len(x_list), dtype=float)
			q_0 = np.full(shape=(len(b), 1), fill_value=0.5)
			for i in range(len(b)):
				x = x_list[i]
				if x < 0.5:
					value = 1
					q_0[i] = value

			
			q_0 = np.append(q_0, np.zeros(shape=(len(b), 1)))

			L_l = np.array([1, 0])
			L_r = np.array([1, 0])


		elif self.type == "lake at rest":
			L = 25
			x_list = [x*L/(x_steps-1) for x in range(x_steps)]
			b = np.zeros(len(x_list), dtype=float)
			q_0 = np.full(shape=(len(b), 1), fill_value=0.5)
			for i in range(len(b)):
				x = x_list[i]
				if 8 < x < 12:
					value = 0.2 - 0.05*(x-10)**2
					b[i] = value
					q_0[i] -= value


			q_0 = np.append(q_0, np.zeros(shape=(len(b), 1)))

			L_l = np.array([1, 0])
			L_r = np.array([1, 0])


		elif self.type == "flat":
			L = 25
			x_list = [x*L/(x_steps-1) for x in range(x_steps)]
			b = np.zeros(len(x_list), dtype=float)
			q_0 = np.full(shape=(len(b), 1), fill_value=0.5)


			q_0 = np.append(q_0, np.zeros(shape=(len(b), 1)))

			L_l = np.array([1, 1])
			L_r = np.array([1, 1])

		
		elif self.type == "gaussian pulse":
			L = 1
			x_list = [x*L/(x_steps-1) for x in range(x_steps)]
			b = np.zeros(len(x_list), dtype=float)
			h_0 = 1.0
			q_0 = np.full(shape=(len(b), 1), fill_value=h_0)

			G_amp = 0.1
			r_0 = 0.1
			x_0 = 0.5

			for i in range(len(b)):
				x = x_list[i]
				value = G_amp * np.exp(-((x-x_0)/r_0)**2)
				q_0[i] += value


			q_0 = np.append(q_0, np.zeros(shape=(len(b), 1)))

			L_l = np.array([0, 1])
			L_r = np.array([0, 1])
		

		elif self.type == "slope":
			L = 5
			x_list = [x*L/(x_steps-1) for x in range(x_steps)]
			b = np.full(len(x_list), fill_value=0.1)
			q_0 = np.full(shape=(len(b), 1), fill_value=0.2)
			for i in range(len(b)):
				x = x_list[i]
				if 2 < x < 3:
					value = (0.1 + 2/10) - x/10
					b[i] = value
				elif x >= 3:
					value = 0
					b[i] = value


			q_0 = np.append(q_0, np.full(shape=(len(b), 1), fill_value=0.1))

			L_l = np.array([5, 1/0.1])
			L_r = np.array([0, 0])


		elif self.type == "moving lake":
			L = 25
			x_list = [x*L/(x_steps-1) for x in range(x_steps)]
			b = np.zeros(len(x_list), dtype=float)
			q_0 = np.full(shape=(len(b), 1), fill_value=0.5)
			for i in range(len(b)):
				x = x_list[i]
				if 8 < x < 12:
					value = 0.2 - 0.05*(x-10)**2
					b[i] = value
					q_0[i] -= value

			#hu = np.array([h*1.0 for h in q_0])
			
			q2 = np.zeros(shape=(len(b), 1))
			#q2[0] = 1

			q_0 = np.append(q_0, q2)

			L_l = np.array([0, 1])
			L_r = np.array([0, 1])


		elif self.type == "gaussian square":
			L = 50
			x_list = [x*L/(x_steps-1) for x in range(x_steps)]
			b = np.zeros(len(x_list), dtype=float)
			q_0 = np.full(shape=(len(b), 1), fill_value=0.5)
			for i in range(len(b)):
				x = x_list[i]
				""" if x < 12:
					value = 0.45
					b[i] = value
					q_0[i] -= value """
				value = 0.45/(1+np.exp(10*(x-12)))
				b[i] = value
				q_0[i] -= value

			q_0 = np.append(q_0, np.zeros(shape=(len(b), 1)))

			G_amp = 0.05
			r_0 = 1
			x_0 = 25

			for i in range(len(b)):
				x = x_list[i]
				value = G_amp * np.exp(-((x-x_0)/r_0)**2)
				q_0[i] += value


			L_l = np.array([0, 1])
			L_r = np.array([0, 1])


		elif self.type == "gaussian lake":
			L = 50
			x_list = [x*L/(x_steps-1) for x in range(x_steps)]
			b = np.zeros(len(x_list), dtype=float)
			q_0 = np.full(shape=(len(b), 1), fill_value=0.21)
			for i in range(len(b)):
				x = x_list[i]
				if 8 < x < 12:
					value = 0.2 - 0.05*(x-10)**2
					b[i] = value
					q_0[i] -= value

			q_0 = np.append(q_0, np.zeros(shape=(len(b), 1)))

			G_amp = 0.05
			r_0 = 1
			x_0 = 25

			for i in range(len(b)):
				x = x_list[i]
				value = G_amp * np.exp(-((x-x_0)/r_0)**2)
				q_0[i] += value


			L_l = np.array([0, 1])
			L_r = np.array([0, 1])


		elif self.type == "gaussian ramp":
			L = 50
			x_list = [x*L/(x_steps-1) for x in range(x_steps)]
			b = np.zeros(len(x_list), dtype=float)
			q_0 = np.full(shape=(len(b), 1), fill_value=0.5)
			for i in range(len(b)):
				x = x_list[i]
				""" if x < 12:
					value = 0.45
					b[i] = value
					q_0[i] -= value """
				value = -0.009*x + 0.45
				b[i] = value
				q_0[i] -= value

			q_0 = np.append(q_0, np.zeros(shape=(len(b), 1)))

			G_amp = 0.05
			r_0 = 1
			x_0 = 25

			for i in range(len(b)):
				x = x_list[i]
				value = G_amp * np.exp(-((x-x_0)/r_0)**2)
				q_0[i] += value


			L_l = np.array([0, 1])
			L_r = np.array([0, 1])


		elif self.type == "gaussian tsunami":
			L = 10000
			x_list = [x*L/(x_steps-1) for x in range(x_steps)]
			b = np.zeros(len(x_list), dtype=float)
			q_0 = np.full(shape=(len(b), 1), fill_value=4500.0)
			for i in range(len(b)):
				x = x_list[i]
				""" if x < 12:
					value = 0.45
					b[i] = value
					q_0[i] -= value """
				value = 4000.0 - 4000/10000 * x
				b[i] = value
				q_0[i] -= value

			q_0 = np.append(q_0, np.zeros(shape=(len(b), 1)))

			G_amp = 5
			r_0 = 100
			x_0 = 5000

			for i in range(len(b)):
				x = x_list[i]
				value = G_amp * np.exp(-((x-x_0)/r_0)**2)
				q_0[i] += value


			L_l = np.array([0, 1])
			L_r = np.array([0, 1])


		elif self.type == "flowing lake":
			L = 250
			x_list = [x*L/(x_steps-1) for x in range(x_steps)]
			b = np.zeros(len(x_list), dtype=float)
			q_0 = np.full(shape=(len(b), 1), fill_value=1.147)
			for i in range(len(b)):
				x = x_list[i]
				if 8 < x < 12:
					value = 0.2 - 0.05*(x-10)**2
					b[i] = value
					q_0[i] -= value
				#q_0[i] = -(1/3)*()


			q_0 = np.append(q_0, np.zeros(len(x_list), dtype=float))

			L_l = np.array([0, 1])
			L_r = np.array([1, 0])


		time_step = L/x_steps*0.01

		return x_list, b, time_step, L_l, L_r, q_0, L





if __name__ == "__main__":
	r = Run(type="flowing lake", max_time=40, integration="RK4")
	
	#r.error_calc()
	#r.disp_last_image(x_steps=200, order_of_accuracy=5)
	#r.run_simulation(x_steps=400, order_of_accuracy=5)
	#r.gaus_error_calc()
	#r.convergence_calc()
	r.flow_error_calc()