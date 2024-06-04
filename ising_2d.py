import numpy as np
import sys
import matplotlib.pyplot as plt


"""
PHYS-E0415
STATISTICAL MECHANICS D 
ALAVA/SALMENJOKI
CODE FOR ISING IN 2D
"""


np.random.seed(444222)
plt.style.use('ggplot')

## initialize a random grid of spins of -1 and 1
def initialize(N):
	return 2*np.random.randint(2, size=(N,N))-1

## metropolis-hastings step to determine a random spin to flip and see if the flip is valid 
def metropolis(grid, beta, H): 
	N = grid.shape[0]
	for i in range(N*N):
		pos = np.random.randint(0,N, size=2)	
		nbrs = grid[(pos[0]+1)%N, pos[1]] + grid[(pos[0]-1)%N, pos[1]] +grid[pos[0], (pos[1]+1)%N] +grid[pos[0], (pos[1]-1)%N] 

		dE = 2*grid[pos[0], pos[1]]*(nbrs+H)

		## flip if dE<0 or with prob exp^(-dE*beta) 
		if dE<0 or np.random.rand()<np.exp(-dE*beta):  
			grid[pos[0], pos[1]] *=-1

	return grid


## compute the energy of the current spin configuration
def energy(grid, H):
	E = 0 
	N = grid.shape[0]
	for x in range(N):
		for y in range(N):
			nbrs = grid[(x+1)%N, y] + grid[(x-1)%N, y] +grid[x, (y+1)%N] +grid[x, (y-1)%N] 
			E -=  (nbrs+H) * grid[x,y]
	return 1.0*E/4 ## avoid overcounting


def magnetization(grid):
	return np.sum(grid)


def plot_system(grid, t,T, H):
	fig = plt.figure(t+1, figsize=(12,8))	
	plt.imshow(np.copy(grid), interpolation='nearest', cmap='binary', vmin=-1, vmax=1, origin='lower')
	plt.title("System at time={}, T={:.2f}, external field H={:.2f}".format(t,T, H))
	plt.grid()

	## uncomment if you want to save the system configurations
	#np.savetxt("configuration_ising_2d_{}_{}_{}_{}.dat".format(grid.shape[0],t,T,H), grid)

	return fig 


def main(arglist):

	snaps = []
	n_temp = 100

	## resolve command line parameters
	if len(arglist)<4 or len(arglist)==5:	
		print( "USAGE:\n    python ising_2d.py [SYSTEMSIZE] [STEPS] [EXTERNALFIELD]\n    python ising_2d.py [SYSTEMSIZE] [STEPS] [EXTERNALFIELD] [TEMPERATURE] [SNAPSHOT_T1]...")
		return
	else:
		N = int(arglist[1])
		steps = int(arglist[2])
		H = float(arglist[3])
		if len(arglist)>4:
			temp = [float(arglist[4])]
			for i in range(len(arglist[5:])):
				snaps.append(int(arglist[5+i]))
		else:
			temp = np.linspace(1.5,3.5,n_temp)	

	## small sanity check on input parameters
	if N<2 or steps<1 or temp[0]<0:
		print("Invalid command line parameters")
		return

	## Energy and other parameters we want to compute
	E = np.zeros(len(temp))
	M = np.zeros(len(temp))
	C = np.zeros(len(temp))
	X = np.zeros(len(temp))

	## parameters to calculate running average (notice that these are averages per spin)
	n1 = 1.0/(steps*N*N)
	n2 = 1.0/(steps*steps*N*N)

	for ii, T in enumerate(temp):
		E1=0
		M1=0
		E2=0
		M2=0
		grid = initialize(N) ## get the initial configuration
		beta = 1.0/T ## k_B = 1  

		
		## first we equilibrate the system 
		## (assumption is that snapshots are wanted here)
		for t in range(steps):
			if t in snaps:
				plot_system(grid, t, T, H)	

			metropolis(grid, beta, H)	

		## then we start to actually collect data, if we aren't just plotting snapshots
		if len(snaps)==0:
			for t in range(steps):
				metropolis(grid, beta, H)	
				tE = energy(grid, H)
				tM = magnetization(grid)
			
				E1 += tE	
				E2 += tE*tE
				M1 += tM			
				M2 += tM*tM
		
			E[ii] = n1*E1	
			M[ii] = n1*M1
			C[ii] = beta*beta*(n1*E2 - n2*E1*E1)
			X[ii] = beta*(n1*M2 - n2*M1*M1)
				
	## then we plot a figure with energy, magnetization, specific heat and susceptibility
	if len(snaps)==0:
		plt.figure(figsize=(12,12))
	
		plt.subplot(2,2,1)
		plt.title('external field H={}'.format(H))
		plt.plot(temp, E, 'ro', markeredgecolor='none', markersize=5)
		plt.xlabel('Temperature')
		plt.ylabel('Energy')
			
		plt.subplot(2,2,2)
		plt.plot(temp, M, 'go', markeredgecolor='none', markersize=5)
		plt.xlabel('Temperature')
		plt.ylabel('Magnetization')

		plt.subplot(2,2,3)
		plt.plot(temp, C, 'mo', markeredgecolor='none', markersize=5)
		plt.xlabel('Temperature')
		plt.ylabel('Specific heat')

		plt.subplot(2,2,4)
		plt.plot(temp, X, 'bo', markeredgecolor='none', markersize=5)
		plt.xlabel('Temperature')
		plt.ylabel('Susceptibility')


		plt.tight_layout()
		## uncomment if you want to save the data
		#np.savetxt("ising_2d_{}_{}_{}.dat".format(N,steps,H), np.array([temp, E, M , C,X]).transpose())

	plt.show()




if __name__=="__main__":
	main(sys.argv)



