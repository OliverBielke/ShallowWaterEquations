#Runs the simulation

import numpy as np
from constants import Constants
from Animation import Animation
from scipy.integrate import RK45

class Simulation:

	def __init__(self, max_time:float, bottom_list:list, time_step:float, L_l:np.ndarray, \
			  L_r:np.ndarray, q_0:np.ndarray, L:float, integration:str, order_of_accuracy:int, \
				type:str):
		#Check
		integration_list = ["RK4", "Euler"]
		if integration not in integration_list:
			raise ValueError("integration must be one of the following:", integration_list)

		#constants
		self.t_max = max_time #The simulation time
		self.t_step = time_step #Time step size
		self.c = Constants(b_underline=np.array(bottom_list), L=L, \
					 OOA=order_of_accuracy) #Defines the constants in a different class
		self.type = type #The simulation type
		
		#Health check
		self.__health_check(check_constants=True)
		
		if integration == "Euler":
			self.t_list = [t*time_step for t in range(int(max_time//time_step +1))]
		elif integration == "RK4":
			self.t_list = []
		self.integration = integration

		self.L_l = L_l #Boundary condition
		self.L_r = L_r #Boundary condition

		#variables
		self.q = q_0
		self.h_list = [q_0.copy()[:self.c.m]]
		self.t = 0
	

	def simulate(self):
		"""Runs the simulation. """
		
		#Euler method
		if self.integration == "Euler":
			for t in self.t_list:
				self.t = t

				#Calculate one step
				self.__Euler_integration_step()
				
				#Save the step
				self.h_list.append(self.q.copy()[:self.c.m])

				if t % 1 < 10**-5 or t % 1 > (1 - 10**-5):
					print("t =", t)
		
				#self.__health_check()
		
		#Runge-Kutta method
		elif self.integration == "RK4":
			#Start the Runge-Kutta
			rk = RK45(fun=self.__q_t, t0=0, y0=self.q, t_bound=self.t_max, \
			 max_step=self.t_step, first_step=self.t_step)
			
			#Iterate over all time steps
			while rk.t < rk.t_bound:
				#One step
				rk.step()

				#Record step
				self.q = rk.y

				#Save the step
				self.h_list.append(self.q.copy()[:self.c.m])
				self.t_list.append(rk.t)
				self.t = rk.t
				
				#print(f"{rk.t / rk.t_bound * 100:.1f}%")
				""" if rk.t % 1 < 10**-5 or rk.t % 1 > (1 - 10**-5):
					print("t =", rk.t) """
				if self.c.m == 1601:
					print(rk.t)

				#self.__health_check()


	def __Euler_integration_step(self) -> None:
		"""Makes one time integration step. """
		self.q += (self.__q_t(None, self.q))*self.t_step #The integration step


	def __q_t(self, t=None, y=None) -> np.ndarray:
		"""Returns the time derivative of q for the current time step. """
		q = self.q
		
		SAT = self.SAT()
		#Calculating the equation except SAT and __q_t
		equation1 = np.dot(self.c.I2_D1, self.__F())
		#Think this is correct order?
		equation2 = np.kron(self.__R(), self.c.H_inv_dot_S)
		equation2 = np.dot(equation2, (q + self.c.b_overline))
		equation = equation1 - equation2 + self.__G()

		return SAT.flatten() - equation
		#return 0 - equation


	def __F(self) -> np.ndarray:
		"""Returns the F matrix given the current q. Output of size 2m x 1. """
		#Definining things used
		m = self.c.m
		g = self.c.g
		q = self.q
		
		part1 = q[m:] #All the q_2

		def one_part2(q_1:float, q_2:float) -> float:
			return q_2**2 / q_1 + (1/2) * g * q_1**2

		part2 = np.array([one_part2(q_1=q[i], q_2=q[m+i]) for i in range(m)]) #The second half
		part2.transpose() #Makes it in to an "standing" vector

		return np.append(part1, part2) #Combine the first and last part


	def __G(self) -> np.ndarray:
		"""Returns G(q), output size of 2m x 1. """

		#Things used many times
		b_underline = self.c.b_underline
		b_double_underline = self.c.b_double_underline
		g = self.c.g
		D1 = self.c.D1
		q = self.q
		
		#The first part of the lower part
		part1 = np.dot(D1, b_underline)
		q1_double_underline = np.diag(q[:self.c.m])
		part1 = np.dot((q1_double_underline + b_double_underline), part1)
		part1 *= g


		#The second part of the lower part
		part2 = np.dot(b_double_underline, b_underline)
		part2 = g/2 * part2
		part2 = np.dot(D1, part2)

		return np.append(self.c.zero, part1 - part2)


	def __A(self, index:int) -> np.ndarray:
		"""Returns the A matrix given the current q and which point to calculate A for. Output of size 2 x 2. """
		#Definining things used
		m = self.c.m
		q = self.q
		g = self.c.g
		q_1 = q[index]
		q_2 = q[m+index]

		#Top half
		part1 = np.array([0, 1]) 

		#Bottom left
		part21 = g*q_1 - q_2**2 / q_1**2 

		#Bottom right
		part22 = 2*q_2/q_1

		#Bottom half
		part2 = np.array([part21, part22])

		return np.array([part1, part2])


	def __R(self) -> np.ndarray:
		"""Returns R(t) as a 2x2 matrix. """
		
		#Things used
		m = self.c.m
		
		#Finding largest eigenvalue of A
		alpha = 0
		for i in range(m):
			plus, minus = self.__A_eigen(i=i) #Getting the eigenvalues

			#If new has larger magnitude than before
			alpha = max(alpha, abs(plus), abs(minus))

		return alpha * self.c.I_2


	def __A_eigen(self, i:int) -> tuple[float]:
		"""Returns the plus and minus eigenvalue of A for a given point. \
			i is the index of the point in q. Output of float, float. """
		#Things used
		g = self.c.g
		m = self.c.m
		q = self.q
		
		#q values
		h = q[i]
		u = q[m+i] / h
		
		#Result
		plus = u + np.sqrt(g*h)
		minus = u - np.sqrt(g*h)

		return plus, minus


	def __g_L(self) -> float:
		"""Returns the g_L(t). """
		if self.type == "lake at rest":
			return 0.5
		elif self.type == "dam break":
			return 1
		elif self.type == "flat":
			return 0.5
		elif self.type == "gaussian pulse":
			return 0
		elif self.type == "slope":
			return 2
		elif self.type == "moving lake":
			return 1
		elif self.type == "gaussian lake":
			return self.q[self.c.m+1]
		elif self.type == "gaussian square":
			return self.q[self.c.m+1]
		elif self.type == "gaussian ramp":
			return self.q[self.c.m+1]
		#*(self.q[0]/self.q[1])
		elif self.type == "gaussian tsunami":
			return self.q[self.c.m+1]
		elif self.type == "flowing lake":
			return 4.42
		else:
			raise ValueError("Wrong")

	def __g_R(self) -> float:
		"""Returns the g_R(t). """
		if self.type == "lake at rest":
			return 0.5
		elif self.type == "dam break":
			return 0.5
		elif self.type == "flat":
			return 0.5
		elif self.type == "gaussian pulse":
			return 0
		elif self.type == "slope":
			return 0
		elif self.type == "moving lake":
			return self.q[-2]
		elif self.type == "gaussian lake":
			return self.q[-2]
		elif self.type == "gaussian square":
			return self.q[-2]
		elif self.type == "gaussian ramp":
			return self.q[-2]
		#*(self.q[self.c.m-1]/self.q[self.c.m-2])
		elif self.type == "gaussian tsunami":
			return self.q[-2]
		elif self.type == "flowing lake":
			#return self.q[self.c.m-2]
			return 1.147
		else:
			raise ValueError("Wrong")


	def SAT(self) -> np.ndarray:
		"""Returns the SAT. Output of size 2m x 1. """
		#Things used
		q = self.q
		Lambda_plus, _ = self.__A_eigen(i=0)
		_, Lambda_minus = self.__A_eigen(i=self.c.m-1)
		
		if Lambda_plus <= 0 or Lambda_minus >= 0:
			raise ValueError("Not subcritical flow")

		W_plus = np.array([[1], [Lambda_plus]])
		W_minus = np.array([[1], [Lambda_minus]])

		#Left part
		left_part = np.dot(self.c.e_LT, q)
		left_part = np.dot(self.L_l, left_part)
		left_part -= self.__g_L() #Everything inside left parenthesis
		inside_par = np.dot(self.L_l, W_plus)
		left_part *= 1 / inside_par.item()
		left_part *= Lambda_plus
		left_part = W_plus * left_part
		left_part = np.dot(self.c.e_L, left_part)
		left_part = - np.dot(self.c.H_bar_inv, left_part)

		#Right part
		right_part = np.dot(self.c.e_RT, q)
		right_part = np.dot(self.L_r, right_part)
		right_part -= self.__g_R() #Everything inside right parenthesis
		inside_par = np.dot(self.L_r, W_minus)
		right_part *= 1 / inside_par.item()
		right_part *= Lambda_minus
		right_part = W_minus * right_part
		right_part = np.dot(self.c.e_R, right_part)
		right_part = np.dot(self.c.H_bar_inv, right_part)
		
		""" new_q = (left_part + right_part).flatten()*self.t_step + q 
		
		if abs(new_q[0]-0.5) > abs(q[0]-0.5) or abs(new_q[self.c.m-1]-0.5) > abs(q[self.c.m-1]-0.5):
			#raise ValueError("SAT too large")
			pass """


		return left_part + right_part


	def __health_check(self, check_constants:bool=False) -> None: #TODO
		"""Checks that everything is correct"""
		if check_constants == True:
			#Check b_overline
			self.__check_b_overline()
			
			#Check b_double_underline
			self.__check_b_double_underline()

			#Check S
			self.__check_S()
		
		elif self.type == "lake at rest" or self.type == "flat":
			self.__check_lake()
			
			
	def __check_lake(self) -> None:
		"""Checks that the lake is correct. """
		
		b_list = [self.c.b_underline[i][0] for i in range(len(self.c.b_underline))]
		h_list = self.h_list[-1] + b_list
		h_tol = 10**-10
		for i in range(len(h_list)):
			if abs(h_list[i] - 0.5) > h_tol:
				#raise ValueError("Lake not at h = 0.5 at index = " + str(i) + ", at t = " \
					 #+ str((len(self.h_list) -1) * self.t_step))
				pass
		
		v_tol = 10**-10
		for i in range(len(h_list)):
			if abs(self.q[i+self.c.m]) > v_tol:
				#raise ValueError("Lake not at v = 0 at index = " + str(i) + ", at t = " \
					 #+ str((len(self.h_list) -1) * self.t_step))
				pass



	def __check_b_overline(self) -> None:
		"Help function to health_check, checks b_overline. "
		
		if len(self.c.b_overline) != 2*len(self.c.b_underline):
			raise ValueError("b_overline has incorrect length. It has len = " + str(len(self.c.b_overline))\
					+ ", but should have len = " + str(2*len(self.c.b_underline)))
		
		for i in range(len(self.c.b_overline)//2):
			if self.c.b_overline[i] != self.c.b_underline[i]:
				raise ValueError("First half of b_overline is not correctly classified. Noticed at index = " + str(i))
			if self.c.b_overline[-1-i] != 0:
				raise ValueError("Second half of b_overline is not correctly classified. Noticed at index = " + str(-1-i))


	def __check_b_double_underline(self) -> None:
		"Help function to health_check, checks b_double_underline. "

		if len(self.c.b_double_underline) != len(self.c.b_underline):
			raise ValueError("b_double_underline has not the same side length as b_underlines length. Has sidelength = " \
					+ str(len(self.c.b_double_underline)) + ", but should have sidelength = " + str(len(self.c.b_underline)) + ".")
		

		if len(self.c.b_double_underline) != len(self.c.b_double_underline[0]):
			raise ValueError("b_double_underline not square. Has shape " + str(len(self.c.b_double_underline)) \
					+ " x " + str(self.c.b_double_underline[0]) + ".")
		

		for row in range(len(self.c.b_double_underline)):
			for col in range(len(self.c.b_double_underline[0])):
				if row == col:
					if self.c.b_double_underline[row][col] != self.c.b_underline[row]:
						raise ValueError("b_double_underline wrong in the diagonal. Noticed at (" + str(row) + ", " + str(col) + ").")
				elif self.c.b_double_underline[row][col] != 0:
					raise ValueError("b_double_underline not zero outside of diagonal. Noticed at (" + str(row) + ", " + str(col) + ").")



	def __check_S(self) -> None:
		"""Help function to health_check, checks S. """
		#S needs to be negative semi-definite according to Mattson (2017) p. 286. 
		S_eigenvalues = np.linalg.eigvalsh(self.c.S)
		if not np.all(S_eigenvalues <= 10**-10): #If any eigenvalue is positive (with some slack)
			boboboob = 1
			#raise ValueError("S is not semi-definite")


if __name__ == "__main__":
	x_steps = 10
	#L = 25
	L = 1
	x_list = [x*L/(x_steps-1) for x in range(x_steps)]
	b = np.zeros(len(x_list), dtype=float)
	q_0 = np.full(shape=(len(b), 1), fill_value=0.5)
	for i in range(len(b)):
		x = x_list[i]
		""" if 8 < x < 12:
			value = 0.2 - 0.05*(x-10)**2
			b[i] = value
			q_0[i] -= value """
		if x < 0.5:
			value = 1
			q_0[i] = value

	
	q_0 = np.append(q_0, np.zeros(shape=(len(b), 1)))

	L_l = np.array([1, 0])
	L_r = np.array([1, 0])

	max_time=0.04
	time_step=0.001
	s = Simulation(max_time=max_time, bottom_list=b, time_step=time_step, \
				L_l=L_l, q_0=q_0, L_r=L_r, L=L, integration="Euler", \
					order_of_accuracy=3, type="lake at rest")
	s.simulate()

	

	b_matrix = [b for _ in s.t_list]

	a = Animation(b_matrix=b_matrix, h_matrix=s.h_list, t_list=s.t_list, x_list=x_list)
	a.animate()