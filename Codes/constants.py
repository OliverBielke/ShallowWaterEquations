# The constants for the 1D simulation

import numpy as np

class Constants:

	def __init__(self, b_underline:np.ndarray, L:float, OOA:int):
		self.b_underline = b_underline.reshape(-1, 1).copy() #The bottom topography
		self.b_double_underline = np.diag(self.b_underline.ravel())

		self.m = len(b_underline) #The amount of x divisions
		self.deltax = L/self.m

		self.I_2 = np.array([[1, 0], [0,1]])

		self.zero = np.zeros(shape=(self.m, 1)) #The 0 array

		self.b_overline = np.append(self.b_underline, self.zero)

		self.e_m = self.zero.copy()
		self.e_m[-1] = 1
		
		self.e_1 = self.zero.copy()
		self.e_1[0] = 1

		self.e_L = np.kron(self.I_2, self.e_1)
		self.e_LT = np.transpose(self.e_L)

		self.e_R = np.kron(self.I_2, self.e_m)
		self.e_RT = np.transpose(self.e_R)

		B_diag = np.zeros(self.m)
		B_diag[0] = -1 
		B_diag[-1] = 1
		self.B = np.diag(B_diag)

		self.g = 9.81 #Gravitational constant same as in paper


		if OOA == 2:
			#H, Q and D matrices (second order case)
			self.Qplus = np.zeros((self.m, self.m)) 
			q0, q1, q2 = -3/2, 2, -1/2
			q11, q12, q21, q22 = -1/4, 5/4, -1/4, -5/4

			#corners in matrix
			self.Qplus[[0, self.m-1], [0, self.m-1]] = q11
			self.Qplus[[0,self.m-2], [1, self.m-1]] = q12
			self.Qplus[[1,self.m-1], [0, self.m-2]] = q21
			self.Qplus[[1,self.m-2], [1, self.m-2]] = q22
			self.Qplus[[1], [2]] = q1 #Upper left (lower right will be handled in inner points)
			self.Qplus[[0,1], [2,3]] = q2 #Upper left (lower right will be handled in inner points)

			#inner points
			for i in range(2,self.m-3+1):
				self.Qplus[i,i] = q0
				self.Qplus[i,i+1] = q1
				self.Qplus[i,i+2] = q2

			h1,h2 = 1/4, 5/4
			H_diag = np.ones(self.m)
			H_diag[[0,self.m-1]] = h1
			H_diag[[1,self.m-2]] = h2


		elif OOA == 3:
			#H, Q and D matrices (third order case)
			self.Qplus = np.zeros((self.m, self.m)) 
			q_m1, q0, q1, q2 = -1/3, -1/2, 1, -1/6
			q11, q12, q21, q22 = -1/12, 3/4, -5/12, -5/12

			#corners in matrix
			self.Qplus[[0, self.m-1], [0, self.m-1]] = q11
			self.Qplus[[0,self.m-2], [1, self.m-1]] = q12
			self.Qplus[[1,self.m-1], [0, self.m-2]] = q21
			self.Qplus[[1,self.m-2], [1, self.m-2]] = q22
			self.Qplus[self.m-2,self.m-3] = q_m1 #Wrong here?
			self.Qplus[1,2] = q1 #Need to add point here?
			self.Qplus[[0,1], [2, 3]] = q2

			#inner points
			for i in range(2,self.m-2):
				self.Qplus[i,i-1] = q_m1
				self.Qplus[i,i] = q0
				self.Qplus[i,i+1] = q1
				self.Qplus[i,i+2] = q2

			h1,h2 = 5/12, 13/12
			H_diag = np.ones(self.m)
			H_diag[[0,self.m-1]] = h1
			H_diag[[1,self.m-2]] = h2


		elif OOA == 5:
			#H, Q and D matrices (fith order case)
			self.Qplus = np.zeros((self.m, self.m)) 
			q_m2, q_m1, q0 = 1/20, -1/2, -1/3
			q1, q2, q3 = 1, -1/4, 1/30
			
			q11, q12, q13, q14 = -1/120, 941/1440, -47/360, -7/480
			q21, q22, q23, q24 = -869/1440, -11/120, 25/32, -43/360
			q31, q32, q33, q34 = 29/360, -17/32, -29/120, 1309/1440
			q41, q42, q43, q44 = 1/32, -11/360, -661/1440, -13/40

			#corners in matrix
			self.Qplus[[0, self.m-1], [0, self.m-1]] = q11
			self.Qplus[[0, self.m-2], [1, self.m-1]] = q12
			self.Qplus[[0, self.m-3], [2, self.m-1]] = q13
			self.Qplus[[0, self.m-4], [3, self.m-1]] = q14

			self.Qplus[[1, self.m-1], [0, self.m-2]] = q21
			self.Qplus[[1, self.m-2], [1, self.m-2]] = q22
			self.Qplus[[1, self.m-3], [2, self.m-2]] = q23
			self.Qplus[[1, self.m-4], [3, self.m-2]] = q24

			self.Qplus[[2, self.m-1], [0, self.m-3]] = q31
			self.Qplus[[2, self.m-2], [1, self.m-3]] = q32
			self.Qplus[[2, self.m-3], [2, self.m-3]] = q33
			self.Qplus[[2, self.m-4], [3, self.m-3]] = q34

			self.Qplus[[3,self.m-1], [0, self.m-4]] = q41
			self.Qplus[[3,self.m-2], [1, self.m-4]] = q42
			self.Qplus[[3,self.m-3], [2, self.m-4]] = q43
			self.Qplus[[3,self.m-4], [3, self.m-4]] = q44

			self.Qplus[[self.m-3, self.m-4], [self.m-5, self.m-6]] = q_m2
			self.Qplus[self.m-4, self.m-5] = q_m1
			self.Qplus[3, 4] = q1
			self.Qplus[[2, 3], [4, 5]] = q2
			self.Qplus[[1, 2, 3], [4, 5, 6]] = q3


			#inner points
			for i in range(4,self.m-4):
				self.Qplus[i,i-2] = q_m2
				self.Qplus[i,i-1] = q_m1
				self.Qplus[i,i] = q0
				self.Qplus[i,i+1] = q1
				self.Qplus[i,i+2] = q2
				self.Qplus[i,i+3] = q3

			h1, h2, h3, h4 = 251/720, 299/240, 41/48, 149/144
			H_diag = np.ones(self.m)
			H_diag[[0, self.m-1]] = h1
			H_diag[[1, self.m-2]] = h2
			H_diag[[2, self.m-3]] = h3
			H_diag[[3, self.m-4]] = h4



		elif OOA == 7:
			#H, Q and D matrices (seventh order case)
			self.Qplus = np.zeros((self.m, self.m)) 
			q_m3, q_m2, q_m1, q0 = 1/105, 1/10, -3/5, -1/4
			q1, q2, q3, q4 = 1, -3/10, 1/15, -1/40
			
			q11, q12, q13 = -265/300272, 1587945773/2432203200, -1926361/25737600
			q14, q15, q16 = -84398989/810734400, 48781961/4864406400, 3429119/202683600

			q21, q22, q23 = -1570125773/2432203200, -26517/1501360, 240029831/486440640
			q24, q25, q26 = 202934303/972881280, 118207/13512240, -231357719/4864406400

			q31, q32, q33 = 1626361/25737600, -206937767/486440640, -61067/750680
			q34, q35, q36 = 49602727/81073440, -43783933/194576256, 51815011/810734400

			q41, q42, q43 = 91418989/810734400, -53314099/194576256, -33094279/81073440
			q44, q45, q46 = -18269/107240, 440626231/486440640, -365711063/1621468800

			q51, q52, q53 = -62551961/4864406400, 799/35280, 82588241/972881280
			q54, q55, q56 = -279245719/486440640, -346583/1501360, 2312302333/2432203200

			q61, q62, q63 = -3375119/202683600, 202087559/4864406400, -11297731/810734400
			q64, q65, q66 = 61008503/1621468800, -1360092253/2432203200, -10677/42896

			#corners in matrix
			self.Qplus[[0, self.m-1], [0, self.m-1]] = q11
			self.Qplus[[0, self.m-2], [1, self.m-1]] = q12
			self.Qplus[[0, self.m-3], [2, self.m-1]] = q13
			self.Qplus[[0, self.m-4], [3, self.m-1]] = q14
			self.Qplus[[0, self.m-5], [4, self.m-1]] = q15
			self.Qplus[[0, self.m-6], [5, self.m-1]] = q16

			self.Qplus[[1, self.m-1], [0, self.m-2]] = q21
			self.Qplus[[1, self.m-2], [1, self.m-2]] = q22
			self.Qplus[[1, self.m-3], [2, self.m-2]] = q23
			self.Qplus[[1, self.m-4], [3, self.m-2]] = q24
			self.Qplus[[1, self.m-5], [4, self.m-2]] = q25
			self.Qplus[[1, self.m-6], [5, self.m-2]] = q26

			self.Qplus[[2, self.m-1], [0, self.m-3]] = q31
			self.Qplus[[2, self.m-2], [1, self.m-3]] = q32
			self.Qplus[[2, self.m-3], [2, self.m-3]] = q33
			self.Qplus[[2, self.m-4], [3, self.m-3]] = q34
			self.Qplus[[2, self.m-5], [4, self.m-3]] = q35
			self.Qplus[[2, self.m-6], [5, self.m-3]] = q36

			self.Qplus[[3,self.m-1], [0, self.m-4]] = q41
			self.Qplus[[3,self.m-2], [1, self.m-4]] = q42
			self.Qplus[[3,self.m-3], [2, self.m-4]] = q43
			self.Qplus[[3,self.m-4], [3, self.m-4]] = q44
			self.Qplus[[3,self.m-5], [4, self.m-4]] = q45
			self.Qplus[[3,self.m-6], [5, self.m-4]] = q46

			self.Qplus[[4,self.m-1], [0, self.m-5]] = q51
			self.Qplus[[4,self.m-2], [1, self.m-5]] = q52
			self.Qplus[[4,self.m-3], [2, self.m-5]] = q53
			self.Qplus[[4,self.m-4], [3, self.m-5]] = q54
			self.Qplus[[4,self.m-5], [4, self.m-5]] = q55
			self.Qplus[[4,self.m-6], [5, self.m-5]] = q56

			self.Qplus[[5,self.m-1], [0, self.m-6]] = q61
			self.Qplus[[5,self.m-2], [1, self.m-6]] = q62
			self.Qplus[[5,self.m-3], [2, self.m-6]] = q63
			self.Qplus[[5,self.m-4], [3, self.m-6]] = q64
			self.Qplus[[5,self.m-5], [4, self.m-6]] = q65
			self.Qplus[[5,self.m-6], [5, self.m-6]] = q66


			#inner points
			for i in range(self.m):
				if i-3 >= 0 and self.Qplus[i,i-3] == 0:
					self.Qplus[i,i-3] = q_m3
				if i-2 >= 0 and self.Qplus[i,i-2] == 0:
					self.Qplus[i,i-2] = q_m2
				if i-1 >= 0 and self.Qplus[i,i-1] == 0:
					self.Qplus[i,i-1] = q_m1
				if self.Qplus[i,i] == 0:
					self.Qplus[i,i] = q0
				if i+1 <= self.m-1 and self.Qplus[i,i+1] == 0:
					self.Qplus[i,i+1] = q1
				if i+2 <= self.m-1 and self.Qplus[i,i+2] == 0:
					self.Qplus[i,i+2] = q2
				if i+3 <= self.m-1 and self.Qplus[i,i+3] == 0:
					self.Qplus[i,i+3] = q3
				if i+4 <= self.m-1 and self.Qplus[i,i+4] == 0:
					self.Qplus[i,i+4] = q4

			h1, h2, h3 = 19087/60480, 84199/60480, 18869/30240
			h4, h5, h6 = 37621/30240, 55031/60480, 61343/60480
			H_diag = np.ones(self.m)
			H_diag[[0, self.m-1]] = h1
			H_diag[[1, self.m-2]] = h2
			H_diag[[2, self.m-3]] = h3
			H_diag[[3, self.m-4]] = h4
			H_diag[[4, self.m-5]] = h5
			H_diag[[5, self.m-6]] = h6


		self.Qminus = -self.Qplus.transpose()

		self.S = (self.Qplus - self.Qminus) / 2

		self.H = self.deltax * np.diag(H_diag)
		self.H_inv = np.linalg.inv(self.H)

		self.H_inv_dot_S = np.dot(self.H_inv, self.S)

		self.Dplus = np.dot(self.H_inv, (self.Qplus + 1/2 * self.B))
		self.Dminus = np.dot(self.H_inv, (self.Qminus + 1/2 * self.B))
		self.D1 = (self.Dplus + self.Dminus) / 2
		
		self.I2_D1 = np.kron(self.I_2, self.D1) #Kronecker product of D_1 and I_2

		self.d_double_underline = np.diag(b_underline)

		H_bar = np.kron(self.I_2, self.H)
		self.H_bar_inv = np.linalg.inv(H_bar)



if __name__ == "__main__":
	b = np.ones(10)
	c = Constants(b,1,5)
	print(c.H)