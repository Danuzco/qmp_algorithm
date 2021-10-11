import qiskit.quantum_info as qi
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
import h5py
import scipy
from scipy import optimize
from datetime import datetime
from datetime import timedelta
import os
cwd = os.getcwd()


class DensityMatrix(qi.DensityMatrix):
    """
    Adds features to the qi.DensityMatrix class
    of the qiskit.quantum_info package:
    1) spectra
    2) labels
    """
    
    def __init__(self, data, dims=None):
        self._labels = None
        super().__init__(data, dims)
    
    def get_labels(self):
        return self._labels
    
    def set_labels(self, given_labels):
        self._labels = given_labels
    
    def spectra(self):
        return np.linalg.eigvalsh(self.data)

    
    
def create_storing_folers():
    """ 
    Creates folders, if they don't exist, 
    to store the data
    """
    path_file_plots = cwd + '\\plots\\'
    path_file_data = cwd + '\\data\\'
    path_file_txtdata = cwd + '\\txtdata\\'
    directories = [path_file_plots, path_file_data, 
                   path_file_txtdata]
    
    created = False
    for pathname in directories:
        try:
            os.mkdir(pathname)
            created = True
        except:
            pass
    if created:
        print("Folders to store data has just been created ...")
        

def partial_trace(rho, antisystem):
    """
    Change the order for which 
    the partial trace is taken 
    in qi.partial_trace
    """
    n = len( rho.dims() )
    lista = [n-i-1 for i in antisystem]
    lista.sort()
    labels = set(range(n)) - set(antisystem)
    sigma = DensityMatrix( qi.partial_trace(rho, tuple(lista) ), 
                          dims= rho.dims()[:len(labels)] )
    sigma.set_labels(list(labels))
    return sigma


def qmp(d, num_of_qudits, prescribed_marginals, rank = 1, prescribed_spectra = [], params = {}):
    """
    Inputs:
    d: local dimension
    num_of_qudits: number of qudits
    prescribed_marginals: The input marginals
    rank: default rank = 1. If prescribed_spectra is an empty list, 
          then the algorithm is an prescribed rank mode.
    params: parameters.
    
    Outputs:
    data: A dictionary containing the found matrix 'rho_n', the marginal distancce 'mdistance',
          the eigenvalues distance 'edistance', the overall distance 'dtotal', the global 
          'gdistance' and the runtime 'runtime'.
    """
    
    
    start = time.time()
    
    if len(prescribed_spectra) > 0:
        print("prescribed spectra mode..")
    else:
        print("prescribed rank mode..")

    print(33*'_')
    print(f'      n     tdist     gdist')
    print(33*'_')
    
    
    rho_gen = params['rho_gen']
    txt_file_name = (cwd + '\\txtdata\\' + '_N' + 
                     str(num_of_qudits) + 'd' + str(d) + 'data.txt')
    
    if params['save_iter_data']:
        try:
            open(txt_file_name,'w').close()
            f = open(txt_file_name,'a')
        except:
            f = open(txt_file_name,'w')
  
    dn, swapper_d = d**num_of_qudits, swapper(d)
    dims = tuple([d for i in range(num_of_qudits)])
    
    if params['x_0'] != None:
        x_0 = params['x_0']
    else:
        if params['seed'] == 'mixed':
            x_0 = qi.DensityMatrix( qi.random_density_matrix(dims))
        if params['seed'] == 'mms':
            x_0 = DensityMatrix(np.identity(dn)/dn, dims)
        if params['seed'] == 'pure':
            x_0 = qi.DensityMatrix( qi.random_statevector(dims))
    
    mdistance, edistance, gdistance, dtotal = [], [], [], []
    number_of_iterations, total_distance, previous_total_distance = 0, 10, 10
    marginal_hsd, eigenvals_distance = 10, 10

    runtime = 0

    while total_distance > params['dtol'] and number_of_iterations < params['max_iter']:
        
        x_0 = impose_marginals( d, num_of_qudits, x_0, 
                               prescribed_marginals, swapper_d )
        
        if len(prescribed_spectra) > 0:
            eigenvals_distance, x_0 = impose_prescribed_eigenvals(x_0, prescribed_spectra)
        else:
            eigenvals_distance, x_0 = impose_rank_constraint(x_0, rank)
            
        x_0 = qi.DensityMatrix(x_0, dims)

        # metrics
        resulting_marginals, marginal_hsd = compute_marginals_distance( x_0, prescribed_marginals, 
                                                                       num_of_qudits)     
        total_distance = np.sqrt(marginal_hsd**2 + eigenvals_distance**2)
        dg = np.linalg.norm( x_0.data - rho_gen.data)
        gdistance.append( np.real( dg  ) ) 
        edistance.append( eigenvals_distance ) 
        mdistance.append( np.real( marginal_hsd ) ) 
        dtotal.append( total_distance )    

        if number_of_iterations % params['iter_to_print'] == 0:
            print( "%7d  %4.5E  %4.5E" %( number_of_iterations + 1, 
                                         total_distance, dg ) )
        # stores in h5 data
        if params['save_h5']:
            save_partial_data(x_0, mdistance, edistance, 
                              gdistance, stype = params['h5_name'])
        # stores txt data
        end = time.time()
        runtime += end - start
        if params['save_iter_data']:
            f.write(f'%i  %1.10f  %s\n'%(number_of_iterations + 1, total_distance, 
                                         time_format(timedelta(seconds = runtime)) ))
            f.flush()
        
        number_of_iterations += 1
        start = time.time()
        
    
    if params['save_iter_data']:
        f.close()
    
    runtime += time.time() - start
    data = {'rho_n':x_0, 'mdistance': mdistance, 
            'edistance':edistance, 'gdistance':gdistance, 
            'tdistance':dtotal, 
            'runtime':time_format(timedelta(seconds = runtime)) }
    
    return data



def accelerated_qmp(d, num_of_qudits, prescribed_marginals, rank = 1,
                    prescribed_spectra = [], params = {}):
    
    """
    Inputs:
    d: local dimension
    num_of_qudits: number of qudits
    prescribed_marginals: The input marginals
    rank: default rank = 1. If prescribed_spectra is an empty list, 
          then the algorithm is an prescribed rank mode.
    params: parameters.
    
    Outputs:
    data: A dictionary containing the found matrix 'rho_n', the marginal distancce 'mdistance',
          the eigenvalues distance 'edistance', the overall distance 'dtotal', the global 
          'gdistance' and the runtime 'runtime'.
    """
    
    start = time.time()
    
    if len(prescribed_spectra) > 0:
        print("prescribed spectra mode..")
    else:
        print("prescribed rank mode..")
    
    print(33*'_')
    print(f'      n     tdist     gdist')
    print(33*'_')
    
    rho_gen = params['rho_gen']
    txt_file_name = (cwd + '\\txtdata\\' + '_N' 
                     + str(num_of_qudits) + 'd' 
                     + str(d) + 'data.txt')
    
    if params['save_iter_data']:
        try:
            open(txt_file_name,'w').close()
            f = open(txt_file_name,'a')
        except:
            f = open(txt_file_name,'w')
  
    dn, swapper_d = d**num_of_qudits, swapper(d)
    dims = tuple([d for i in range(num_of_qudits)])
    
    if params['x_0'] != None:
        x_0 = params['x_0']
    else:
        if params['seed'] == 'mixed':
            x_0 = qi.DensityMatrix( qi.random_density_matrix(dims))
        if params['seed'] == 'mms':
            x_0 = DensityMatrix(np.identity(dn)/dn, dims)
        if params['seed'] == 'pure':
            x_0 = qi.DensityMatrix( qi.random_statevector(dims))
    
    mdistance, edistance, gdistance, dtotal = [], [], [], []
    number_of_iterations, total_distance, previous_total_distance = 0, 10, 10
    marginal_hsd, eigenvals_distance, threshold = 10, 10, 1e-1

    alfa, alfa_n = params['alfa'], 1
    mu, bt = params['mu'], params['bt']
    beta = alfa_n**2 + bt

    runtime = 0
    if not params['accelerated']:
        alfa, beta, mu = 1, 0, 0 

    while total_distance > params['dtol'] and number_of_iterations < params['max_iter']:
        
        x_0 = accelerated_impose_marginals( d, num_of_qudits, x_0, 
                                           prescribed_marginals, swapper_d, 
                                           alfa_n, alfa, mu, beta)
        if len(prescribed_spectra) > 0:
            eigenvals_distance, x_0 = impose_prescribed_eigenvals(x_0, 
                                                      prescribed_spectra)
        else:
            eigenvals_distance, x_0 = impose_rank_constraint(x_0, rank)
        
        x_0 = qi.DensityMatrix(x_0, dims)
        resulting_marginals, marginal_hsd = compute_marginals_distance( x_0, 
                                               prescribed_marginals,num_of_qudits)     
        total_distance = np.sqrt(marginal_hsd**2 + eigenvals_distance**2)
        dg = np.linalg.norm( x_0.data - rho_gen.data)
        gdistance.append( np.real( dg  ) ) 
        edistance.append( eigenvals_distance ) 
        mdistance.append( np.real( marginal_hsd ) ) 
        dtotal.append( total_distance )    
        
        if params['accelerated']:
            alfa_n = 1/(1e-5*number_of_iterations + 1)**alfa  
            beta = alfa_n**2 + bt
        if number_of_iterations % params['iter_to_print'] == 0:
            print( "%7d  %4.5E  %4.5E" %( number_of_iterations + 1, 
                                         total_distance, dg ) )
        # stores in h5 data
        if params['save_h5']:
            save_partial_data(x_0, mdistance, edistance, 
                              gdistance, stype = params['h5_name'])
        # stores txt data
        end = time.time()
        runtime += end - start
        if params['save_iter_data']:
            f.write(f'%i  %1.10f  %s\n'%(number_of_iterations + 1, total_distance, 
                                         time_format(timedelta(seconds = runtime)) ))
            f.flush()
        number_of_iterations += 1
        start = time.time()
        
    if params['save_iter_data']:
        f.close()
    
    runtime += time.time() - start
    data = {'rho_n':x_0, 'mdistance': mdistance, 
            'edistance':edistance, 'gdistance':gdistance, 
            'tdistance':dtotal, 
            'runtime':time_format(timedelta(seconds = runtime)) }
    
    return data


def swapper(d):
    """
    returns the Swap operator of dimension d
    """
    p = 0
    Id = np.identity(d)
    for i in range(d):
        for j in range(d):
            v = np.outer(Id[:,i],Id[:,j])
            u = np.transpose(v)
            p += np.kron(v,u)      
    return p


def kron(*matrices):
    
    m1, m2, *ms = matrices
    m3 = np.kron(m1, m2)
  
    for m in ms:
        m3 = np.kron(m3, m)
    
    return m3


def Pj( in_label, marginal, dl, num_of_qudits, swapper_d ):
    """
    dl: local dimension
    marginal: reduced system with labels given in the tuple "in_label"
    
    returns: The tensor product of the terms in the operator Q in the proper order
    
    """
    
    label = in_label
    n = num_of_qudits - len( marginal.dims() ) 
    dims = tuple( [ dl for i in range( num_of_qudits ) ] )
    swapped_matrix = kron( marginal.data, np.identity( dl**n ) )
    
    all_labels = [ i for i in range( num_of_qudits ) ]
    right_labels = [ i for i in range( list( label )[-1] + 1, num_of_qudits ) ]
    left_labels = [ i for i in range( list( label )[0] ) ]
    
    if left_labels + list( label ) + right_labels == all_labels:
        nl  = list(label)[0] 
        nr = num_of_qudits - nl  - len(label)
        Il, Ir = np.identity( dl**nl ), np.identity( dl**nr )
        swapped_matrix = qi.DensityMatrix( kron( Il, marginal.data, Ir ) , dims )
        swapped_matrix = swapped_matrix/swapped_matrix.trace()       
        return swapped_matrix
    else:
        nl = list(label)[0] 
        nr = num_of_qudits - nl  - len(label)
        Il, Ir = np.identity( dl**nl ), np.identity( dl**nr )
        swapped_matrix = kron( Il, marginal.data, Ir )
        label = tuple( left_labels + list( label ) )
        
    length = len( label )
    remaining = tuple( [ i for i in range( length ) ] )
    
    while length > 0 and label != remaining:
        
        last = label[-1]
        numOfswapps = np.abs( last  - length ) 
        l1, l2 = length - 1, num_of_qudits - ( length + 1 )
        I1, I2 = np.identity( dl**l1 ), np.identity( dl**l2 )
        gate = kron( I1, swapper_d, I2 ) 
        swapped_matrix = gate @ swapped_matrix @ gate
        
        for i in range( numOfswapps ):
            l1, l2 = l1 + 1, l2 - 1 
            I1, I2 = np.identity( dl**l1 ), np.identity( dl**l2 )
            gate = kron( I1, swapper_d, I2 )
            swapped_matrix = gate @ swapped_matrix @ gate

        label = tuple( list( label[:-1] ) )
        length = len( label )
        remaining = tuple( [ i for i in range( length ) ] )
    
    swapped_matrix = qi.DensityMatrix( swapped_matrix, dims )
    swapped_matrix = swapped_matrix/swapped_matrix.trace()
    
    return swapped_matrix


def impose_prescribed_eigenvals(x_0, prescribed_spectra):
    
    dn, _ = x_0.data.shape
    rank = len(prescribed_spectra)
    prescribed_spectra = sorted(prescribed_spectra)
    prescribed_spectra = np.concatenate(( np.zeros((dn-rank)), 
                                         prescribed_spectra ))
    eigenvalues, eigenvects = np.linalg.eigh( x_0.data )    
    eigenvals_distance = np.linalg.norm( eigenvalues - prescribed_spectra )
    rhof = eigenvects @ np.diag( prescribed_spectra ) @ np.conjugate( eigenvects.T )
    
    return eigenvals_distance, rhof/np.trace(rhof)


def impose_rank_constraint(x_0, rank):
    
    dn, _ = x_0.data.shape
    eigenvalues, eigenvects = np.linalg.eigh( x_0.data )
    
    a, b = np.zeros((dn-rank)), eigenvalues[ dn - rank: ]
    if len(eigenvalues[ eigenvalues < 0]) <= rank:
        eigenvals_distance = np.linalg.norm( eigenvalues[ eigenvalues < 0] )
        eigenvalues[ eigenvalues < 0] =  - eigenvalues[ eigenvalues < 0]
    else:
        eigenvals_distance = np.linalg.norm( eigenvalues[:-rank] )
   
    rhof = eigenvects @ np.diag( np.concatenate((a,b)) ) @ np.conjugate( eigenvects.T )

    return eigenvals_distance, rhof/np.trace(rhof)


def impose_marginals(d, num_of_qudits, x_0, prescribed_marginals, swapper_d):

    dn = d**num_of_qudits
    all_systems = set( list( range(num_of_qudits)) )

    for l in list( prescribed_marginals.keys() ):
        antisys = tuple( all_systems - set( l ) )
        tr_rho0_I = partial_trace( x_0 , list( antisys ) )
        x_0 = (x_0  + Pj( l, prescribed_marginals[l], d, num_of_qudits, swapper_d ) -  
               Pj( l, tr_rho0_I, d, num_of_qudits, swapper_d )  )

    return x_0


def accelerated_impose_marginals(d, num_of_qudits, x_0, prescribed_marginals, 
                                 swapper_d, alfa_n, alfa, mu, beta):

    all_systems = set( list( range(num_of_qudits)) )
    dims = tuple( [ d for i in range( num_of_qudits ) ] )
    dn = d**num_of_qudits

    for l in list( prescribed_marginals.keys() ):
        antisys = tuple( all_systems - set( l ) )
        tr_rho0_I = partial_trace( x_0, list(antisys) )
        d_0 = ( Pj( l, prescribed_marginals[l], d, num_of_qudits, swapper_d ).data 
               - Pj( l, tr_rho0_I, d, num_of_qudits, swapper_d ).data )/alfa
        d_1 = d_0 + beta*d_0
        y_0 = x_0.data + alfa*d_1
        x_0 = mu*alfa_n*x_0 + (1 - mu*alfa_n)*y_0
        x_0 = qi.DensityMatrix( x_0, dims )

    return x_0/x_0.trace()


def compute_marginals_distance(rho0, prescribed_marginals,  num_of_qudits):
    
    all_systems = set( list( range(num_of_qudits)) )
    marginal_hsd = 0
    projected_marginals = {}
    
    for l in list( prescribed_marginals.keys() ):
        antisys = tuple( all_systems - set(l) )
        projected_marginals[l] = partial_trace( rho0, list(antisys) )
        marginal_hsd += np.linalg.norm( projected_marginals[l].data 
                                       - prescribed_marginals[l].data)**2
    norm = len( list( prescribed_marginals.keys() ) )
    marginal_hsd = np.sqrt(marginal_hsd/norm)
    
    return projected_marginals, marginal_hsd



def simul_data(d, num_of_qudits, labels_marginals, dtype = "pure" ):

    dn = d**num_of_qudits
    dims = tuple([d for i in range(num_of_qudits)])
    
    if dtype == "AME":
        rho_gen = np.identity(dn)/dn
        rho_gen = qi.DensityMatrix(rho_gen, dims)
    elif dtype == "mixed":
        rho_gen = qi.random_density_matrix(dims).data 
        rho_gen = qi.DensityMatrix(rho_gen, dims)
    elif dtype == "pure":
        rho_gen = qi.DensityMatrix(qi.random_statevector(dn), dims)

    marginals = {}
    all_systems = set( list( range(num_of_qudits)) )

    for s in labels_marginals:
        tracedSystems = tuple( all_systems - set( s ) )
        if len(tracedSystems) > 0:
            marginals[s] = partial_trace(rho_gen, list(tracedSystems))
        else:
            marginals[s] = partial_trace(rho_gen, list(s))
            
    return marginals, rho_gen



def time_format(runtime):

    t = str(runtime)
    h,mi,s = t.split(':')
    s = str(np.round(np.float(s),2))
    
    f,dec = s.split('.')
    if int(f) < 10:
        s = '0' + s

    if int(dec) < 10:
        dec = dec + '0'
    
    return h + ':' + mi + ':' + s



def plot_data(d, num_of_qudits, edistance, mdistance, 
              gdistance, runtime, params = {} ):
    
    cwd = os.getcwd()
    a, h = 7.024, 4.82
  
    plt.figure(figsize=(a,h), dpi = 100 )
    plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.plot( list( range( len(edistance) ) ), edistance, 
             label = r'$\mathcal{D}_{\lambda}$', 
             linestyle='-', linewidth = 1.3,
             markersize = 3.6, mfc='none',  
             color='royalblue')
    plt.plot(  list( range( len(mdistance) ) ), mdistance, 
             label = r'$\mathcal{D}_{M}$', 
             linestyle='--', linewidth = 1.3, 
             markersize = 3.6, mfc='none', 
             color='tab:red')
    if params['plot_global']:
        plt.plot(  list( range( len(gdistance) ) ), gdistance, 
                 label = r'$\mathcal{D}_{G}$', 
                 linestyle='dashdot', linewidth = 1.3, 
                 markersize = 3.6, mfc='none', 
                 color='forestgreen')
        
    plt.title(f'N = {num_of_qudits} ,  d = {d}  ( runtime = {time_format(runtime)})')
    plt.yscale('log')
    plt.xlabel("n", fontsize = 10)

    plt.legend( loc= 'lower left', fontsize = 12)
    
    plt.xlim(0,  list( range( len(edistance) ) )[-1] )
    plt.tick_params(direction='out', axis='y', 
                    which='minor', colors='black')
    
    if params['save_plot']:
        name = ( params['name'] + '_N' + str(num_of_qudits) 
                + 'd' + str(d)  + datetime.today().strftime('_%Y-%m-%d')
                + '.png')
        path_file = cwd + '\\plots\\' + name
        plt.savefig(path_file, format='png',bbox_inches='tight')

    plt.show()


    
def save_partial_data(rho0, mdistance, edistance, gdistance, stype = 'pure'):
    
    Data = {'mhsd': np.array( mdistance), 'edist': np.array(edistance),
            'gdist': np.array(gdistance), 'rho_n': rho0.data}
    
    N = len( rho0.dims() )
    local_dim, *_ = rho0.dims()
    
    file_name = ( cwd + '\\data\\' + stype + 
                 "_rhoN"+ str(N) + "d" 
                 + str(local_dim) + ".h5" )
    try:
        with h5py.File(file_name,'w') as f:
            for l in list( Data.keys() ):
                dtset = f.create_dataset( l, data = Data[l], 
                                         compression="gzip", 
                                         compression_opts=9 )
                f.flush()
    except:
        with h5py.File(file_name,'x') as f:
            for l in list( Data.keys() ):
                dtset = f.create_dataset( l, data = Data[l], 
                                         compression="gzip", 
                                         compression_opts=9 )
                f.flush()
    else:
        with h5py.File(file_name,'a') as f:
            for l in list( Data.keys() ):
                try:
                    del f[l]
                    dtset = f.create_dataset( l, data = Data[l], 
                                             compression="gzip", 
                                             compression_opts=9 )
                except:
                    dtset = f.create_dataset( l, data = Data[l], 
                                             compression="gzip", 
                                             compression_opts=9 )
                f.flush()
                
                
                
def load_h5_data(d, num_of_qudits, stype = 'pure'):
    
    try:
        file_name = (cwd + '\\data\\' + stype 
                     + "_rhoN"+ str(num_of_qudits) 
                     + "d" + str(d) + ".h5")
        g1 = h5py.File(file_name, 'a')
    except:
        file_name = (cwd + '\\data\\' + 'copy_' 
                     + stype + "_rhoN"
                     + str(num_of_qudits) 
                     + "d" + str(d) + ".h5")
        g1 = h5py.File(file_name, 'a')
    
    data = {'mhsd':np.array( g1['mhsd']),
            'edist':np.array(g1['edist']),
            'gdist':np.array(g1['gdist']),
            'rho_n': np.array(g1['rho_n'])}
    
    return data
    
    
    

def compare_plots(d, num_of_qudits, gdistance, gdistance_accel,
                  gdistance_accel1, runtime, runtime_accel, 
                  runtime_accel1, params = {} ):
    
    cwd = os.getcwd()
    a, h = 7.024, 4.82
    plt.figure(figsize=(a,h), dpi = 100 )
    numOfIter = len(gdistance)
    numOfIterAccel = len(gdistance_accel)
    numOfIterAccel1 = len(gdistance_accel1)
    plt.plot(list(range(numOfIter)), gdistance, 
             label = f'Alg (2)  ({time_format(runtime)})', 
             linestyle='-', linewidth = 1.3, markersize = 3.6, mfc='none',  
             color='m')
    plt.plot(list(range(numOfIterAccel)), gdistance_accel, 
             label = f'Alg (3)  ({time_format(runtime_accel)})', 
             linestyle=':', linewidth = 1.3, markersize = 3.6, mfc='none', 
             color='tab:blue')
    
    plt.plot(list(range(numOfIterAccel1)), gdistance_accel1, 
             label = f'Alg (3)  ({time_format(runtime_accel1)})', 
             linestyle='--', linewidth = 1.3, markersize = 3.6, mfc='none', 
             color='tab:green')
        
    plt.title(f'N = {num_of_qudits} ,  d = {d}')
    plt.yscale('log')
    plt.xlabel("n", fontsize = 10)
    
    NofI = max([numOfIter,numOfIterAccel,numOfIterAccel1 ])
    
    dg = {numOfIter:gdistance,numOfIterAccel:gdistance_accel,
          numOfIterAccel1:gdistance_accel1}

    plt.legend( loc= 'upper right', fontsize = 12)
    
    plt.xlim(0, list(range(NofI))[-1])    
    plt.ylim(dg[NofI][-1], dg[NofI][0])

    plt.tick_params(direction='out', axis='y', 
                    which='minor', colors='black')
    
    if params['save_plot']:
        name = ( params['name'] + '_N' + str(num_of_qudits) 
                + 'd' + str(d)  + datetime.today().strftime('_%Y-%m-%d') 
                + '.png')
        path_file = cwd + '\\plots_comparison\\' + name
        plt.savefig(path_file, format='png',bbox_inches='tight')

    plt.show()

