#Animating the water

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Animation:

	def __init__(self, b_matrix:list[np.ndarray], h_matrix:list[np.ndarray], \
			  t_list:list, x_list:list):
		self.b_matrix = b_matrix #Bottom height
		self.w_matrix = [b + h for b, h in zip(b_matrix, h_matrix)] #Water height (b + h)
		self.h_matrix = h_matrix
		self.x_list = x_list #x-coord for the b/h-lists
		self.t_list = t_list #Time list

	
	def animate(self):
		fig, ax = plt.subplots()
		#ax.set_xlim(self.x_list[0], self.x_list[-1]) #x-limits
		ax.set_xlim(self.x_list[0], self.x_list[-1]) #x-limits
		max_y = np.max(self.w_matrix)
		ax.set_ylim(0, max_y*1.2) #y-limits
		#ax.set_aspect('equal')

		b_line, = ax.plot([], [], label="Bottom", color="brown")
		w_line, = ax.plot([], [], label="Water", color="blue")
		title = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center")
		ax.legend()


		def init(): #First step
			#Set lines
			b_line.set_data([], [])
			w_line.set_data([], [])
			return b_line, w_line


		def one_frame(i): #One time step
			#Update lines
			b_line.set_data(self.x_list, self.b_matrix[i])
			w_line.set_data(self.x_list, self.w_matrix[i])
			
			#Update timer
			title.set_text(f'Time: {self.t_list[i]:.3f} s')

			return b_line, w_line


		delta_t = self.t_list[-1] / len(self.t_list)
		interval_ms = delta_t #convert to ms

		""" ani = FuncAnimation(fig=fig, func=one_frame, frames=len(self.t_list), \
					  interval=interval_ms, blit=True) """
		
		ani = FuncAnimation(fig=fig, func=one_frame, frames=len(self.t_list), interval=0.001)

		plt.show()


	def plot_last(self):
		
		test_list = []
		for i in range(len(self.x_list)):
			a = 1
			b = self.b_matrix[-1][i] - 4.42**2/(2*9.81*self.h_matrix[-1][0]**2) - self.h_matrix[-1][0]
			d = 4.42**2/(2*9.81)
			test_list.append(self.__cubic_solve(a=a, b=b, d=d) + self.b_matrix[-1][i])
		
		fig, ax = plt.subplots()
		#ax.set_xlim(self.x_list[0], self.x_list[-1]) #x-limits
		ax.set_xlim(self.x_list[0], 25) #x-limits
		max_y = np.max(self.w_matrix[-1])
		#ax.set_ylim(0, max_y*1.2) #y-limits
		ax.set_ylim(0, 3.2) #y-limits
		#h_compare = np.full(shape=(len(self.b_matrix[-1]), 1), fill_value=0.5)

		plt.plot(self.x_list, self.b_matrix[-1], label="Bottom", color="brown")
		
		#plt.plot(self.x_list, self.w_matrix[0], label="Initial", color="orange", linestyle="--")
		#plt.plot(self.x_list, h_compare, label="Water", color="blue")
		# self.w_matrix = [(w - 1)/4 for w in self.w_matrix]
		# plt.plot(self.x_list, self.w_matrix[0], color="blue")
		# self.w_matrix = [w + 0.22/4 for w in self.w_matrix]
		# plt.plot(self.x_list, self.w_matrix[len(self.w_matrix)//4], color="blue")
		# self.w_matrix = [w + 0.22/4 for w in self.w_matrix]
		# plt.plot(self.x_list, self.w_matrix[2*len(self.w_matrix)//4], color="blue")
		# self.w_matrix = [w + 0.22/4 for w in self.w_matrix]
		# plt.plot(self.x_list, self.w_matrix[3*len(self.w_matrix)//4], color="blue")
		# self.w_matrix = [w + 0.22/4 for w in self.w_matrix]
		plt.plot(self.x_list, self.w_matrix[-1], color="blue", label="Water")
		plt.plot(self.x_list, test_list, color="yellow", label="Exact", linestyle=":")
		
		plt.xlabel("x [m]")
		plt.ylabel("height [m]")

		plt.legend()

		plt.show()


		fig, ax = plt.subplots()
		#ax.set_xlim(self.x_list[0], self.x_list[-1]) #x-limits
		ax.set_xlim(self.x_list[0], 25) #x-limits
		max_y = np.max(self.w_matrix[-1])
		ax.set_ylim(0, 3.2) #y-limits

		plt.plot(self.x_list, self.b_matrix[0], label="Bottom", color="brown")

		plt.plot(self.x_list, self.w_matrix[round(len(self.b_matrix)*1.5/self.t_list[-1])], color="blue", label="Water")
		
		plt.xlabel("x [m]")
		plt.ylabel("height [m]")

		plt.legend()

		plt.show()


		fig, ax = plt.subplots()
		#ax.set_xlim(self.x_list[0], self.x_list[-1]) #x-limits
		ax.set_xlim(self.x_list[0], 25) #x-limits
		max_y = np.max(self.w_matrix[-1])
		ax.set_ylim(0, 3.2) #y-limits

		plt.plot(self.x_list, self.b_matrix[0], label="Bottom", color="brown")

		plt.plot(self.x_list, self.w_matrix[round(len(self.b_matrix)*4.2/self.t_list[-1])], color="blue", label="Water")
		
		plt.xlabel("x [m]")
		plt.ylabel("height [m]")

		plt.legend()

		plt.show()


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



if __name__ == "__main__":
	pass