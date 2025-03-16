import numpy as np
import scipy.sparse as spp
import scipy.optimize as spo
import scipy.integrate as spi
from math import nan
import warnings
import dis

def euler_step(f,t,x,p,step_size,tfin):
    """
    Description:
    ------------

    Performs a single Euler method step 
    
    Args:
    -----
    
    f (function): Single function f(t,x,p) which outputs a single numpy array, with one element for the value in each dimension
    t (float): Current value of t
    x (numpy.ndarray): Array containing the current x values for each dimension
    p (numpy.ndarray): Numpy array of parameters in the ODE, given in the order they appear
    step_size (float): Current step size for t
    tfin (float): Final value of t
    
    Returns:
    --------
    
    t_new (float): New value of t
    x_new (numpy.ndarray): Array containing the new x value for each dimension
    
    """
    # calculating new t
    t_new=t+step_size
    # correction of last timestep to land exactly on tfin
    t_new,step_size=final_timestep_correct(t,t_new,tfin,step_size)
    # calculating new x
    x_new=x+step_size*f(t,x,p)  
    return t_new,x_new

def solve_to(f,p,t0,x0,tfin,delta_max=1e-4,solver='rk4'):
    """
    Description:
    ------------
    Solves ODE for a time range, with the default method being Euler's method, the user can specify the use of 4th order Runge Kutta
    
    Args:
    -----
    
    f (function): Single function f(t,x,p) which outputs a single numpy array, with one element for the value in each dimension
    p (numpy.ndarray): Numpy array of parameters in the ODE, given in the order they appear
    t0 (float): Initial value of t
    x0 (numpy.ndarray): Numpy array of initial values of x for each dimension
    tfin (float): Final value of t
    delta_max (float): Maximum timestep. Defaults to 1e-4
    solver (str): Type of solver used. Options are 'euler' or 'rk4', with solver='rk4' being the default value.
    
    Returns:
    --------
    
    t_vals (numpy.ndarray): Array of all t values
    x_vals (numpy.ndarray): Array of all x values, where each row is solution for one dimension
    """
    # checking f is a function
    if not callable(f):
        raise TypeError("f must be a function")
    # checking all values are numeric
    if not isinstance(t0,(int,float)):
        raise TypeError("t0 must be a numeric value (int or float)")
    if not all(isinstance(val,(np.integer,np.floating)) for val in x0):
        raise TypeError("All elements of x0 must be numeric values (int or float)")
    # checking that x0 is a numpy array
    if not isinstance(x0, np.ndarray):
        raise TypeError("Initial condition x0 must be a numpy array")
    # checking output of f is a numpy array
    if not isinstance(f(t0,x0,p),np.ndarray):
        raise TypeError("Output of function f must be a numpy array")
    # checking dimension of initial conditions
    if len(x0)!=len(f(t0,x0,p)):
        raise ValueError(f'Dimension of x0 ({len(x0)}) does not match dimension of f ({len(f(t0,x0,p))})')
    # choosing appropriate solver
    if solver.lower()=='rk4':
        solve_step=rk4_step
    elif solver.lower()=='euler':
        solve_step=euler_step
    # returning error for incorrect input
    else:
        raise ValueError(f"'{solver}' is an invalid input string for 'solver'. Please enter 'euler' or 'rk4'.")
    # calculating the number of steps to take
    steps=abs(int(np.ceil(((tfin-t0)/delta_max))))
    # pre-allocating x and t arrays
    x_vals=np.zeros([len(x0),steps+1])
    t_vals=np.zeros([steps+1])
    # setting initial values of arrays
    t_vals[0]=t0
    x_vals[:,0]=x0 
    # setting initial values for x and t 
    t_current=t0
    x_current=x0
    # looping through each timestep and calculating solution at each stage
    for time_step in range(1,steps+1): 
        # performing update step
        t_new,x_new=solve_step(f,t_current,x_current,p,delta_max,tfin)
        # setting newly calculated x and t in the corresponding arrays
        t_vals[time_step]=t_new 
        x_vals[:,time_step]=x_new
        # setting the newly calculated x and t to be the current x and t
        t_current=t_new
        x_current=x_new
    return t_vals,x_vals

def rk4_step(f,t,x,p,step_size,tfin):
    """
    Description:
    ------------

    Single Classical Runge Kutta (RK4) method step 

    Args:
    -----
    
    f (function): Single function f(t,x,p) which outputs a single numpy array, with one element for the value in each dimension
    t (float): Current value of t
    x (numpy.ndarray): Array containing the current x values for each dimension
    p (numpy.ndarray): Numpy array of parameters in the ODE, given in the order they appear
    step_size (float): Current step size
    tfin (float): Final value of t
    
    Returns:
    --------

    t_new (float): New value of t
    x_new (numpy.ndarray): Array containing the new x value for each dimension
    """
    # calculating new t
    t_new=t+step_size
    # correction of last timestep to land exactly on tfin
    t_new,step_size=final_timestep_correct(t,t_new,tfin,step_size)
    # calculation of Runge-Kutta stages
    k1=f(t,x,p)
    k2=f(t+(step_size/2),x+step_size*(k1/2),p)
    k3=f(t+(step_size/2),x+step_size*(k2/2),p)
    k4=f(t+step_size,x+step_size*k3,p)
    # calculating new x
    x_new=x+(step_size/6)*(k1+2*k2+2*k3+k4)  
    return t_new,x_new

def is_equilibrium(x_values, threshold=1e-6):
    """
    Description:
    ------------

    Checks whether the solution the limit cycle has found is a limit cycle or an equilibrium

    Args:
    -----
    
    x_values (numpy.ndarray): Numpy array of the solved limit cycle solution in t 
    threshold (float): Value which defines the limit between being classed as an equilibrium or not
    
    
    Returns:
    --------

    bool: True if the array is deemed an equilibrium, False otherwise.
    """    
    # calculating the difference between each consecutive element
    differences = np.abs(np.diff(x_values))
    # finding the max difference between consecutive elements
    max_difference = np.max(differences)
    return max_difference < threshold

def final_timestep_correct(t,t_new,tfin,step_size):
    """
    Description:
    ------------

    Checks whether the new timestep goes past the end time point, and corrects it accordingly

    Args:
    -----
    
    t (float): Current value of t
    t_new (float): New value of t
    tfin (float): Final value of t
    step_size (float): Current step size for t
    
    
    Returns:
    --------

    t_new (float): New value of t after step size corrected, or original t_new value if step size was not corrected
    step_size (float): Corrected final step size, or original step size if it was not corrected
    """  
    # checking if new timestep goes past finishing timestep
    if t_new>tfin:
        # correcting t_new
        t_new=tfin
        # correcting final step size
        step_size=tfin-t
        return t_new,step_size
    else:
        return t_new,step_size

def lim_cyc_solve(f,p,t0,in_guess,delta_max=1e-2,phase_condition=None,xtol=1.49012e-08,solver='rk4'):
    """
    Description:
    ------------

    Function which finds and returns limit cycle for n-dimensional ODEs 
    
    Args:
    -----
    
    f (function): Single function f(t,x,p) which outputs a single numpy array, with one element for the value in each dimension
    p (numpy.ndarray): Numpy array of parameters in the ODE, given in the order they appear
    t0 (float): Initial value of t
    in_guess (numpy.ndarray): Numpy array of initial guess of x for each dimension, and a final element whose value is the initial guess for the time period of the limit cycle
    delta_max (float): Maximum timestep. Defaults to 1e-2
    phase_condition (function): Function defining the phase condition to be used, where the output of the function will be set to zero. Default is dx/dt(0)=0 
    xtol (float): Tolerance for root solver. Default is 1.49012e-08
    solver (str): Type of solver used. Options are 'euler' or 'rk4', with solver='rk4' being the default value.
    
    Returns:
    --------

    lim_cyc_sol: An instance of lim_cyc_sol with the following attributes:
        - t_vals (numpy.ndarray): Array of all t values for solved limit cycle
        - x_vals (numpy.ndarray): Array of all x values in the solved limit cycle, where each row is the next dimension's solution
        - true_x0 (numpy.ndarray): Array of the initial conditions of the limit cycle and it's period
        - solvet_int (bool): True if the solution array is deemed a limit cycle, False otherwise
        - solver_message (str): Message explaining the output of the solver
    """
    # checking f is a function
    if not callable(f):
        raise TypeError("f must be a function")
     # checking all values are numeric
    if not isinstance(t0,(int,float)):
        raise TypeError("t0 must be a numeric value (int or float)")
    if not all(isinstance(val,(np.integer,np.floating)) for val in in_guess):
        raise TypeError("All elements of x0 must be numeric values (int or float)")
    # checking that x0 is a numpy array
    if not isinstance(in_guess, np.ndarray):
        raise TypeError("Initial guess must be a numpy array")
    # checking output of f is a numpy array
    if not isinstance(f(t0,in_guess[:-1],p),np.ndarray):
        raise TypeError("Output of function f must be a numpy array")
    # setting u to be the initial guess for the root finder
    u=in_guess
    # setting the default phase condition if the user didn't define one
    phase_condition=default_phase_condition(f,phase_condition)
    # defining the function to find the root of 
    g=lambda u: lim_cyc_func(f,p,t0,u,delta_max,solver,phase_condition)
    # running the root finding method on the above function
    true_x0,_,ier,mesg=spo.fsolve(g,u,full_output=True,xtol=xtol) 
    # finding the limit cycle for the solved initial conditions and time period
    t_vals,x_vals=solve_to(f,p,t0,true_x0[:-1],true_x0[-1]+t0,delta_max,solver='euler')
    # returning error message to user if solver failed to converge
    if ier!=1:
        solver_message='Scipy solver failed to converge. Solver message: '+mesg
        solver_int=False
    else:
        # determining if solution is limit cycle or equilibrium
        if is_equilibrium(x_vals):
            solver_message='Solver converged to equilibrium'
            solver_int=False
        else:
            solver_message='Solver converged to limit cycle'
            solver_int=True
    return lim_cyc_sol(t_vals,x_vals,true_x0,solver_int,solver_message)

class lim_cyc_sol:
    def __init__(self,t_vals,x_vals,true_x0,solver_int,solver_message):
        self.t_vals=t_vals
        self.x_vals=x_vals
        self.x0=true_x0
        self.solver_int=solver_int
        self.solver_message=solver_message

def default_phase_condition(f,phase_condition):
    """
    Description:
    ------------

    Function which defines the phase condition as dx/dt(0)=0 if none is provided
    
    Args:
    -----
    
    f (function): Single function f(t,x,p) which outputs a single numpy array, with one element for the value in each dimension
    phase_condition (function): Function defining the phase condition to be used, where the output of the function will be set to zero. Default is dx/dt(0)=0 
    
    Returns:
    --------

    phase_condition (fun): Users original phase condition, or if none is provided the default phase condition of dx/dt(0)=0
    """
    # checking if the user provdided a phase condition
    if phase_condition is None:
        # providing default phase condition if not
        def default_phase_condition(t,x,p):
            return f(t,x,p)[0]
        return default_phase_condition
    else:
        # returning user defined phase condition if they provided one
        return phase_condition

def lim_cyc_func(f,p,t0,u,delta_max,solver,phase_condition):
    """
    Description:
    ------------

    Function which is set to 0 by 'lim_cyc_solve' to find limit cycle 
    
    Args:
    -----
    
    f (function): Single function f(t,x,p) which outputs a single numpy array, with one element for the value in each dimension
    p (numpy.ndarray): Numpy array of parameters in the ODE, given in the order they appear
    t0 (float): Initial value of t
    u (numpy.ndarray): Numpy array of the current guess for x in each dimension, where the final element is the current guess of the period
    delta_max (float): Maximum timestep
    phase_condition (function): Function defining the phase condition to be used, where the output of the function will be set to zero. Default is dx/dt(0)=0 
    
    Returns:
    --------

    numpy.ndarray: Array containing the residuals of the equations to be optimized to zero, where the last element is the phase condition
    """
    # initial guess for x dimensions
    in_guess=u[:-1]
    # initial time period guess
    T=u[-1]
    # solving with current guess in x dimensions and time period
    _,x1=solve_to(f,p,t0,in_guess,t0+T,delta_max,solver)
    # finding the final value of x in the solve
    x1=x1[:,-1]
    # setting phase condition 
    phi=np.array([phase_condition(t0,in_guess,p)])
    # defining the output which is to be optimised to 0
    return np.concatenate([x1-in_guess,phi])

def num_cont(f,p,x0,p_vary=0,stepsize=0.05,max_steps=201,method='nat',xtol=1.49012e-08,args=None,solve_for='equilibria'):
    """
    Description:
    ------------

    Function which performs numerical continuation with user specified preferences 

    If using solve_for = 'lim_cyc', provide arguments in the args tuple in the following order: args = (t0,delta_max,solver,phase_condition). Also note
    that the function provdied as f should be the ODE to solve for limit cycles
    
    Args:
    -----
    
    f (function): Function f(u,p,*args) which is to be solved for at each parameter step.
    p (numpy.ndarray): Numpy array of the initial parameters for the ODE, given in the order they appear
    x0 (nump.ndarray): Initial guess for solution at first parameter value
    p_vary (int): Index of the parameter which is to be varied in the p array, default=0
    stepsize (float): Stepsize for the parameter value, default=0.05
    max_steps (int): Max number of steps in parameter value, default=201
    method (str): Numerical continuation method used. Options are 'nat' and 'arc', with method='nat' being the default option
    xtol (float): Tolerance for root solver. Default is 1.49012e-08
    args (tuple): Tuple containing the extra arguments to f in the order they are passed to the function
    solve_for (str): Type of solve at each parameter value. If 'equilibria', then the equilibria of f are found at each parameter value, if 'lim_cyc', then 
                     limit cycles of f are found at each parameter value, and if 'custom', then the solutions to the user defined f function are returned at 
                     each parameter value
    

    Returns:
    --------

    p_vals (numpy.ndarray): Array of all parameter where the system was solved
    sols (numpy.ndarray): Array containing the soltuion at each parameter value
    """
    # checking user max steps
    if max_steps<1:
        raise ValueError('Max steps should be a positive integer')
    # defining function to be solved for if user chose a supported option
    if solve_for=='equilibria':
        # checking length of intitial guess
        if len(x0)!=len(f(nan,x0,p)):
            raise ValueError(f'Dimension of x0 ({len(x0)}) does not match dimension of f ({len(f(nan,x0,p))})')
        # defining function to be sovled
        g=lambda u,p: f(nan,u,p)
    elif solve_for=='lim_cyc':
        # checking length of initial guess
        if len(x0)!=len(f(nan,x0,p))+1:
            raise ValueError(f'Dimension of x0 ({len(x0)}) should be equal to dimension of system +1, where the extra element is the initial guess for the period of the limit cycle.')
        # defining the provided args
        t0,delta_max,solver,phase_condition=args
        # defining phase condition if user failed to provide one
        phase_condition=default_phase_condition(f,phase_condition)
        # defining function to be sovled
        g=lambda u,p: lim_cyc_func(f,p,t0,u,delta_max,solver,phase_condition)
    # defining user supplied function as function to be solved for if user chooses this option
    elif solve_for=='custom':
        g= lambda u,p: f(u,p,*args)
    else:
        raise ValueError(f"'{solve_for}' is an invalid input string for 'solve_for'. Please enter 'nat' or 'arc'.") 
    # changing the data type of the parameter array to float to allow for non-integer parameter values
    p=p.astype(float)
    # choosing appropriate continuation method and solving
    if method=='nat':
        p_vals,sols=nat_cont(g,p,x0,p_vary,stepsize,max_steps,xtol)
    elif method=='arc':
        p_vals,sols=arc_cont(g,p,x0,p_vary,stepsize,max_steps,xtol)
    else:
        raise ValueError(f"'{method}' is an invalid input string for 'method'. Please enter 'nat' or 'arc'.")    
    return p_vals,sols

def nat_cont(g,p,x0,p_vary,stepsize,max_steps,xtol):
    """
    Description:
    ------------

    Function which performs natural paramter continuation 
    
    Args:
    -----
    
    f (function): Function f(u,p,*args) which is to be solved for at each parameter step.
    p (numpy.ndarray): Numpy array of the initial parameters for the ODE, given in the order they appear
    x0 (nump.ndarray): Initial guess for solution at first parameter value
    p_vary (int): Index of the parameter which is to be varied in the p array
    stepsize (float): Stepsize for the parameter value 
    max_steps (int): Max number of steps in parameter value
    xtol (float): Tolerance for root solver. Default is 1.49012e-08
    
    Returns:
    --------

    p_vals (numpy.ndarray): Array of all parameter where the system was solved
    sols (numpy.ndarray): Array containing the solution at each parameter value
    """
    # setting the array containing the parameter values where the system is solved - rounding to avoid floating point errors
    start_point = round(p[p_vary],10)
    end_point = round(p[p_vary]+stepsize*max_steps,10)
    p_vals=np.arange(start_point,end_point,stepsize)
    # setting the size of the solution array 
    sols=np.zeros([len(x0),p_vals.size])
    # setting the initial u to be x0
    u=x0
    # looping through the predefined parameter values
    for idx,p_val in enumerate(p_vals):
        # updating the varying parameter
        p[p_vary]=p_val
        # solving at current parameter
        u_sol,ier,mesg=nat_step(g,u,p,xtol)
        if ier!=1:
            print(f'Solver failed to converge at paramter value {p_val}. Appending final iteratinon to solution array. Solver message: ', mesg)
        # storing the solution in the solution array
        sols[:,idx]=u_sol
        # updating initial guess to previous solution
        u=u_sol
    return p_vals,sols

def nat_step(g,u,p,xtol):
    """
    Description:
    ------------
    
    Function which performs a single step of natural paramter continuation 
    
    Args:
    -----
    
    f (function): Function f(u,p,*args) which is to be solved for at each parameter step.
    u (numpy.ndarray): Initial guess for the solution at the current parameter values
    p (numpy.ndarray): Current parameter values
    xtol (float): Tolerance for root solver. Default is 1.49012e-08
    
    Returns:
    --------

    u_sol (numpy.ndarray): Solution of f at the current parameter values
    """
    # root finding using the RHS of the provided ODE
    u_sol,_,ier,mesg=spo.fsolve(lambda u: g(u,p),u,full_output=1,xtol=xtol)
    return u_sol,ier,mesg


def arc_step(f,v0,v1,p,p_vary,xtol):
    """
    Description:
    ------------

    Function which performs a single step of pseudo-arclength parameter continuation 
    
    Args:
    -----
    
    f (function): Function f(u,p,*args) which is to be solved for at each parameter step.
    v0 (numpy.ndarray): Augmented solution (parameter value,solution) from two iterations ago
    v1 (numpy.ndarray): Augmented solution (parameter value,solution) from last iteration
    p (numpy.ndarray): Current parameter values
    p_vary (int): Index of the parameter which is to be varied in the parameter array
    xtol (float): Tolerance for root solver. Default is 1.49012e-08
    
    Returns:
    --------

    v_sol (numpy.ndarray): Augmented solution (parameter value,solution) for the current iteration
    """
    # finding the secant using the previous two augmented solutions
    sec=v1-v0
    # predicting the new augmented solution
    v_new=v1+(sec) 
    # defining the function to find the root of
    def g(v):
        # varying the appropriate parameter in the parameter vector
        p[p_vary]=v[0]
        # output which should be 0. First argument is the pseudo-arclength equation, and the second the output of f
        return np.concatenate([np.array([np.dot(v-v_new,sec)]),f(v[1:],p)])
    # setting the initial guess for this iteration
    v=v_new
    # solving for current iteration
    v_sol,_,ier,mesg=spo.fsolve(g,v,full_output=1,xtol=xtol)
    return v_sol,ier,mesg

def arc_cont(f,p,x0,p_vary,stepsize,max_steps,xtol):
    """
    Description:
    ------------

    Function which performs pseudo-arclength continuation 
    
    Args:
    -----
    
    f (function): Function f(u,p,*args) which is to be solved for at each parameter step.
    p (numpy.ndarray): Numpy array of the initial parameters for the ODE, given in the order they appear
    x0 (nump.ndarray): Initial guess for solution
    p_vary (int): Index of the parameter which is to be varied in the parameter array
    stepsize (float): Stepsize for the parameter value 
    max_steps (int): Max number of steps in parameter value
    xtol (float): Tolerance for root solver. Default is 1.49012e-08
    
    Returns:
    --------
    
    p_vals (numpy.ndarray): Array of all parameters where the system was solved
    sols (numpy.ndarray): Array containing the solution at each parameter value
    """
    # predifining size of array for parameter values
    p_vals=np.zeros(max_steps+2)
    # predifining size of array for solutions
    sols=np.zeros([len(x0),max_steps+2])
    # finding initial solution using x0 as an initial guess
    x0,ier,mesg=nat_step(f,x0,p,xtol)
    if ier!=1:
        print('Solver failed to converge at initial guess. Consider changing x0 or xtol. Solver message: ',mesg)
    # forming v0 by concatenating initial solution and parameter value
    v0=np.concatenate([np.array([p[p_vary]]),x0])
    # varying parameter by stepsize to calculate second initial solution
    p[p_vary]+=stepsize
    # using a natual parameter continuation step to calculate second initial solution
    u1,ier,mesg=nat_step(f,x0,p,xtol)
    if ier!=1:
        print('Solver failed to converge at initial guess. Consider changing x0 or xtol. Solver message: ',mesg)
    # forming v1 by concatenating new solution and parameter value
    v1=np.concatenate([np.array([p[p_vary]]),u1])
    # setting first two elements of both arrays to corresponding elements from v0 and v1
    p_vals[:2]=np.array([v0[0],v1[0]])
    sols[:,:2]=np.array([v0[1:],v1[1:]]).T
    # finding the number of solutions specified by the user
    for idx in range(max_steps):
        # finding augmented solution at current iteration
        v_sol,ier,mesg=arc_step(f,v0,v1,p,p_vary,xtol)
        if ier!=1:
            print(f'Solver failed to converge at step {idx}. Appending final iteratinon to solution array. Solver message: ', mesg)
        # assigning appropriate elements of v to corresponding arrays
        p_vals[idx+2]=v_sol[0]
        sols[:,idx+2]=v_sol[1:]
        # updating v0 and v1 values
        v0=v1
        v1=v_sol
    return p_vals,sols

def check_argument(func,index):
    """
    Description:
    ------------

    Checks whether the function uses the variable defined by argument

    Args:
    -----
    
    func (function): Function whose arguments are being checked
    index (int): Index of the argument being checked
    
    
    Returns:
    --------

    bool: True if the function output depends on the argument, False otherwise.
    """  
    # retrieving bytecode instructions for given function
    bytecode_instructions = dis.get_instructions(func)
    # retrieving name of second argument
    second_argument_name = func.__code__.co_varnames[index]
    # Check if the instruction is a load operation and the argument value matches the name of the second argument
    for instruction in bytecode_instructions:
        if instruction.opname.startswith('LOAD_') and instruction.argval == second_argument_name:
            # return true if argument is used
            return True
    # otherwise return false
    return False

def set_LHS(D,P,dx,A1,A2):
    """
    Description:
    ------------

    Function which sets the LHS of the matrix equation when the system is solved using linear algebra
    
    Args:
    -----
    
    D (float): Value of the coefficient of the second order derivative
    P (float): Value of the coefficient of the first order derivative
    dx (float): Distance between each grid point
    A1 (numpy.ndarray or scipy.sparse.spmatrix): Array representing the A matrix for the second order derivative equations
    A2 (numpy.ndarray or scipy.sparse.spmatrix): Array representing the the A matrix for the first order derivative equations

    Returns:
    --------
    
    LHS (numpy.ndarray or scipy.sparse.spmatrix): Array representing the LHS of the matrix equation
    """
    LHS=((D/(dx)**2)*A1)+(P/(2*dx))*A2
    return LHS
def set_RHS(D,P,dx,b_vec_1,b_vec_2,q_eval):
    """
    Description:
    ------------

    Function which sets the RHS of the matrix equation when the system is solved using linear algebra
    
    Args:
    -----
    
    D (float): Value of the coefficient of the second order derivative
    P (float): Value of the coefficient of the first order derivative
    dx (float): Distance between each grid point
    b_vec_1 (numpy.ndarray): Array representing the b vector for the second order derivative equations
    b_vec_2 (numpy.ndarray): Array representing the b vector for the first order derivative equations

    Returns:
    --------
    
    RHS (numpy.ndarray): Array representing the RHS of the matrix equation
    """
    RHS=-b_vec_1*(D/(dx)**2)-b_vec_2*(P/(2*dx))-q_eval
    return RHS

def set_thomas_diags(D,P,dx,diags):
    """
    Description:
    ------------

    Function which sets the diagonals for the LHS when using the Thomas algorithm
    
    Args:
    -----
    
    D (float): Value of the coefficient of the second order derivative
    P (float): Value of the coefficient of the first order derivative
    dx (float): Distance between each grid point
    diags (tuple): Tuple containing the diagonals of the LHS matrices in the form of individual arrays

    Returns:
    --------
    
    sup_diag (numpy.ndarray): Array representing super diagonal of the LHS system
    main_diag (numpy.ndarray): Array representing main diagonal of the LHS system
    sub_diag (numpy.ndarray): Array representing sub diagonal of the LHS system
    """
    # unpacking the arrays tuple
    A_sup_diag,A_main_diag,A_sub_diag,A2_sup_diag,A2_main_diag,A2_sub_diag=diags
    # setting each diagonal
    sup_diag=(D/(dx)**2)*A_sup_diag+(P/2*dx)*A2_sup_diag
    main_diag=(D/(dx)**2)*A_main_diag+(P/2*dx)*A2_main_diag
    sub_diag=(D/(dx)**2)*A_sub_diag+(P/2*dx)*A2_sub_diag
    return sup_diag,main_diag,sub_diag

def fin_diff(a,b,boundary_conds,N=100,D=1,P=1,q=None,p=None,in_guess=None,tol=None,solver='scipy',format='dense'):
    """
    Description:
    ------------

    Function which performs finite differencing to solve BVPs for second order ODEs of the form D(d^2u/dx^2) + P(du/dx) + q(x,u,p) = 0
    
    Note that for Robin boundary conditions, the input ('R',A,B) represents condition du/dx(a) = A - B*u(a), noting the negative application of the constant B
    
    Args:
    -----
    
    a (float): x value of the start of the domain
    b (float): x value of the end of the domain
    boundary_conds (list of two tuples): A list containing two tuples, where each tuple represents the boundary condition for one end of the domain.
                                         Each tuple should contain two elements: the type of boundary condition ('D', 'N', or 'R') and the corresponding
                                         boundary value. For Robin ('R') conditions, you may enter two values, where the first corresponds to the constant,
                                         and the second to the (negated) coefficient of u(x)
    N (int): Number of steps taken in discretisation of space (number of grid points - 1), default=100
    D (float): Value of the coefficient of the second order derivative, default=1
    P (float): Value of the coefficient of the first order derivative, default=1
    q (function): Function q(x,u,p) which represents the source term in the ODE. Defualt is no source term
    p (numpy.ndarray): Numpy array of the parameters contained in the source term, given in the order they appear. Default is no parameters
    tol (float): Tolerance for the scipy root function if this method is selected
    in_guess (numpy.ndarray): Numpy array of an initial guess for the solution if required. Default is a zero vector
    solver (str): Solver used. Options are 'thomas', 'numpy' and 'scipy', with 'scipy' as default. Note you cannot solve nonlinear equations with numpy or the thomas algorithm.
    format (str): Format of matrices used. Default is 'dense'. Note that format='sparse' is only applicable for solver='scipy'
    
    Returns:
    --------
    
    x (numpy.ndarray): Numpy array of the x values that the solution is calculated at
    sol (numpy.ndarray): Numpy array of the solution at the specified x values
    """  
    # if the constants are given as arrays (to be compatible with shooting), take the first element
    if isinstance(P, np.ndarray):
        P=P[0]
    if isinstance(D, np.ndarray):
        D=D[0]
    # calculating the distance between grid points
    dx=(b-a)/N
    # finding matrices and x vector to solve
    diags,b_vec_1,b_vec_2,x=grid(N,a,b,boundary_conds,dx,format)
    # making source term zeros if none is provided
    if q is None:
        def q(x,u,p):
            return np.zeros(len(x))
    else:
        if len(q(x,x,p))!=len(x):
            raise ValueError(f'Output of source function needs to be a numpy array of size {len(x)}')
    if solver!='scipy':
        if format=='sparse':
            raise ValueError(f"Can only solve sparse matrix problems with scipy solver, not {solver} solver")
            # solving with scipy if source term is nonlinear
        if check_argument(q,1):
            warnings.warn(f"Cannot solve nonlinear equation with {solver} solver. Solving with Scipy instead.",stacklevel=2)
            # solving with scipy
            sol=sp_solve(x,dx,D,P,q,p,in_guess,boundary_conds,tol)
        else:
            # evaluating q at grid points and parameter values
            q_eval=q(x,nan,p)
            # forming RHS of matrix equation to solve - not formimg LHS incase thomas algorithm is selected and raw diagonals can be used
            RHS=set_RHS(D,P,dx,b_vec_1,b_vec_2,q_eval)
            if solver=='thomas':
                # setting the diagonals for the LHS
                sup_diag,main_diag,sub_diag=set_thomas_diags(D,P,dx,diags)
                # solving with the thomas algorithm
                sol=thomas_solve(sup_diag,main_diag,sub_diag,RHS)
            else:
                # reforming LHS matrices
                A1,A2=reform_matrices(diags,format)
                # forming LHS of matrix equation
                LHS=set_LHS(D,P,dx,A1,A2)
                # solving with numpy
                sol=np.linalg.solve(LHS,RHS)
    else:
        if format=='sparse':
            # evaluating q at grid points and parameter values
            q_eval=q(x,nan,p)
            # reforming LHS matrices
            A1,A2=reform_matrices(diags,format)
            # forming RHS of matrix equation to solve
            RHS=set_RHS(D,P,dx,b_vec_1,b_vec_2,q_eval)
            # forming LHS of matrix equation to solve
            LHS=set_LHS(D,P,dx,A1,A2)
            # solving using scipy sparse matrices
            sol=spp.linalg.spsolve(LHS,RHS)
        else:
            # solving with scipy root function
            sol=sp_solve(x,dx,D,P,q,p,in_guess,boundary_conds,tol)
    # appending value at 'a' to the solution if LH boundary cond is Dirichlet
    if boundary_conds[0][0]=='D':
        sol=np.concatenate([np.array([boundary_conds[0][1]]),sol])
    # appending value at 'b' to the solution if RH boundary cond is Dirichlet
    if boundary_conds[1][0]=='D':
        sol=np.concatenate([sol,np.array([boundary_conds[1][1]])])
    # reforming the complete x vector
    x=np.linspace(a,b,N+1)
    return x,sol

def set_length(boundary_conds,N):
    """
    Description:
    ------------

    Function which sets the length of the grid for matrix equation formation
    
    Args:
    -----
    
    boundary_conds (list of two tuples): A list containing two tuples, where each tuple represents the boundary condition for one end of the domain
    N (int): Number of steps taken in discretisation of space (number of grid points - 1)

    Returns:
    --------
    
    int: Length of the x solution vector for matrix equation
    """
    if boundary_conds[0][0]=='D' and boundary_conds[1][0]=='D':
        return N-1
    elif boundary_conds[0][0]=='D' or boundary_conds[1][0]=='D':
        return N
    else:
        return N+1
def initialise_arrays(length):
    """
    Description:
    ------------

    Function which sets the diagonal vectors and RHS vectors of the matrix equation to the correct length
    
    Args:
    -----
    
    length (int): Length of the x solution vector for matrix equation

    Returns:
    --------
    
    diags (tuple): Tuple containing the base diagonals of the LHS matrices in the form of individual arrays
    b_vec (numpy.ndarray): Array representing the base b vector for the second order derivative equations
    b_vec_2 (numpy.ndarray): Array representing the base b vector for the first order derivative equations
    """
    # setting the base vectors for the A matrices' diagonals
    A_sup_diag=np.ones(length-1)
    A_main_diag=-2*np.ones(length)
    A_sub_diag=np.ones(length-1)
    A2_sup_diag=np.ones(length-1)
    A2_main_diag=np.zeros(length)
    A2_sub_diag=-1*np.ones(length-1)
    # setting the base vectors for the b vectors
    b_vec=np.zeros(length)
    b_vec_2=np.zeros(length)
    # storing all diags arrays in the diags tuple
    diags=(A_sup_diag,A_main_diag,A_sub_diag,A2_sup_diag,A2_main_diag,A2_sub_diag)
    return diags,b_vec,b_vec_2

def set_values(boundary_conds,dx,diags,b_vec,b_vec_2,x):
    """
    Description:
    ------------

    Function which modifies the base A diagnoals and b vectors to contain the correct values
    
    Args:
    -----
    
    boundary_conds (list of two tuples): A list containing two tuples, where each tuple represents the boundary condition for one end of the domain
    dx (float): Distance between each grid point
    diags (tuple): Tuple containing the base diagonals of the LHS matrices in the form of individual arrays
    b_vec (numpy.ndarray): Array representing the base b vector for the second order derivative equations
    b_vec_2 (numpy.ndarray): Array representing the base b vector for the first order derivative equations
    x (numpy.ndarray): The base x vector of grid points where the solution is to be calculated

    
    Returns:
    --------
    
    diags (tuple): Tuple containing the diagonals of the LHS matrices in the form of individual arrays
    b_vec (numpy.ndarray): Array representing the b vector for the second order derivative equations. Note that if either boundary condition was specified
                             as a function, the b_vector will contain the coefficient which multiplies the function evaluation in the corresponding position 
                             of the vector.
    b_vec_2 (numpy.ndarray): Array representing the b vector for the first order derivative equations
    x (numpy.ndarray): The final x vector of grid points where the solution is to be calculated
    """
    # unpacking the diags tuple
    A_sup_diag,A_main_diag,A_sub_diag,A2_sup_diag,A2_main_diag,A2_sub_diag=diags
    if boundary_conds[0][0]=='D':
        # if the first boundary condition is Dirichlet
        if not callable(boundary_conds[0][1]):
            # setting b vector elements
            b_vec[0]=boundary_conds[0][1]
            b_vec_2[0]=-boundary_conds[0][1]
        else:
            # setting the coefficient to later muliply by function evaluation
            b_vec[0]=1
        # adjusting the x vector
        x=x[1:]
    else:
        # if the first boundary condition is not Dirichlet (Neumann or Robin)
        if not callable(boundary_conds[0][1]):
            # setting b vector elements
            b_vec[0]=-2*boundary_conds[0][1]*dx
            b_vec_2[0]=2*boundary_conds[0][1]*dx
        else:
            # setting the coefficient to later muliply by function evaluation
            b_vec[0]=-2*dx
        # setting A diagonal elements
        A_sup_diag[0]=2
        A2_sup_diag[0]=0
        A2_sub_diag[0]=0
        if boundary_conds[0][0]=='R':
            # setting A diagonal elements
            A_main_diag[0]=-2*(1-boundary_conds[0][2]*dx)
            A2_main_diag[0]=-2*(boundary_conds[0][2]*dx)
    if boundary_conds[1][0]=='D':
        # if the second boundary condition is Dirichlet
        if not callable(boundary_conds[1][1]):
            # setting b vector elements
            b_vec[-1]=boundary_conds[1][1]
            b_vec_2[-1]=boundary_conds[1][1]
        else:
            # setting the coefficient to later muliply by function evaluation
            b_vec[-1]=1
        x=x[:-1]
    else:
        # if the second boundary condition is not Dirichlet (Neumann or Robin)
        if not callable(boundary_conds[1][1]):
            # setting b vector elements
            b_vec[-1]=2*boundary_conds[1][1]*dx
            b_vec_2[-1]=2*boundary_conds[1][1]*dx
        else:
            # setting the coefficient to later muliply by function evaluation
            b_vec[-1]=-2*dx
        # setting A diagonal elements
        A_sub_diag[-1]=2
        A2_sup_diag[-1]=0
        A2_sub_diag[-1]=0
        if boundary_conds[1][0]=='R':
            # setting A diagonal elements
            A_main_diag[-1]=-2*(1+boundary_conds[1][2]*dx)
            A2_main_diag[-1]=-2*(boundary_conds[1][2]*dx)
    # storing all diags arrays in the diags tuple
    diags=A_sup_diag,A_main_diag,A_sub_diag,A2_sup_diag,A2_main_diag,A2_sub_diag
    return diags,b_vec,b_vec_2,x
def reform_matrices(diags,format):
    """
    Description:
    ------------

    Function which reforms the A matrices from their diagonals
    
    Args:
    -----
    
    diags (tuple): Tuple containing the diagonals of the LHS matrices in the form of individual arrays
    format (str): Format of matrices used
    
    Returns:
    --------
    
    A1 (numpy.ndarray or scipy.sparse.spmatrix): Array (sparse or dense) representing the A matrix for the second order derivative equations
    A2 (numpy.ndarray or scipy.sparse.spmatrix): Array (sparse or dense) representing the the A matrix for the first order derivative equations
    """
    # unpacking the diags tuple
    A_sup_diag,A_main_diag,A_sub_diag,A2_sup_diag,A2_main_diag,A2_sub_diag=diags
    if format=='sparse':
        # returning 'csc' format scipy sparse matrices
        A=spp.diags([A_sup_diag,A_main_diag,A_sub_diag],offsets=[1,0,-1],format='csc')
        A2=spp.diags([A2_sup_diag,A2_main_diag,A2_sub_diag],offsets=[1,0,-1],format='csc')
    else:
        # returning numpy arrays
        A=np.diag(A_sup_diag,1)+np.diag(A_sub_diag,-1)+np.diag(A_main_diag)
        A2=np.diag(A2_sup_diag,1)+np.diag(A2_sub_diag,-1)+np.diag(A2_main_diag)
    return A,A2

def time_dependent_conds(b_vec,boundary_conds):
    """
    Description:
    ------------

    Function which defines the b vector as a function if the user specified a time-dependent boundary condition
    
    Args:
    -----
    
    b_vec (numpy.ndarray): Array representing the b vector for the second order derivative equations. Note that if either boundary condition was specified
                             as a function, the b_vector will contain the coefficient which multiplies the function evaluation in the corresponding position 
                             of the vector.
    boundary_conds (list of two tuples): A list containing two tuples, where each tuple represents the boundary condition for one end of the domain
    
    Returns:
    --------
    
    b_vec or b_vec_func (numpy.ndarray or function): Returns the originally provided b_vec if neither boundary condition was time-dependent, or returns b_vec as a function 
                                       of time which accounts for the user's time dependent boundary condition, if one was provided.
    """
    # checking if either boundary condition has been supplied as a function
    if not callable(boundary_conds[0][1]) and not callable(boundary_conds[1][1]):
        # if not, returning the original b vector
        return b_vec
    else:
        # defining the appropriate b vector function
        def b_vec_func(t):
            b=b_vec
            if callable(boundary_conds[0][1]):
                # multiplying by the relevant coefficient
                b[0]=b[0]*boundary_conds[0][1](t)
            if callable(boundary_conds[1][1]):
                # multiplying by the relevant coefficient
                b[-1]=b[-1]*boundary_conds[1][1](t)
            return b 
        return b_vec_func

def grid(N,a,b,boundary_conds,dx,format):
    """
    Description:
    ------------

    Function which creates the grids needed for finite differencing and pde solvers
    
    Args:
    -----
    
    N (int): Number of steps to approximate the solution at
    a (float): x value of the start of the domain
    b (float): x value of the end of the domain
    boundary_conds (list of two tuples): A list containing two tuples, where each tuple represents the boundary condition for one end of the domain
    dx (float): Distance between each grid point
    format (str): Format of matrices used
    
    Returns:
    --------
    
    diags (tuple): Tuple containing the diagonals of the LHS matrices in the form of individual arrays
    b_vec (numpy.ndarray): Array representing the b vector for the second order derivative equations. Note that if either boundary condition was specified
                             as a function, the b_vector will contain the coefficient which multiplies the function evaluation in the corresponding position 
                             of the vector.
    b_vec_2 (numpy.ndarray): Array representing the b vector for the first order derivative equations
    x (numpy.ndarray): The final x vector of grid points where the solution is to be calculated
    """
    # defining the length of the solution vector
    length=set_length(boundary_conds,N)
    # setting the diagonals of the matrices and the b vectors
    diags,b_vec,b_vec_2=initialise_arrays(length)
    # setting the base grid point vector
    x=np.linspace(a,b,N+1)
    # producing the final diags, and b and x vectors
    diags,b_vec,b_vec_2,x=set_values(boundary_conds,dx,diags,b_vec,b_vec_2,x)
    # defining b_vec as the appropriate function if there are time dependent boundary conditions
    b_vec=time_dependent_conds(b_vec,boundary_conds)
    return diags,b_vec,b_vec_2,x

def sp_solve(x,dx,D,P,q,p,in_guess,boundary_conds,tol):
    """
    Description:
    ------------

    Function which solves a BVP using scipys root function    

    Note this is the only solver option which is able to deal with nonlinear source terms in the Poisson equation
    
    Args:
    -----

    x (numpy.ndarray): The x vector of grid points where the solution needs to be calculated
    dx (float): Distance between each grid point
    D (float): Value of the coefficient of the second order derivative
    P (float): Value of the coefficient of the first order derivative
    q (fun): function q(x,u,p) which represents the source term in the ODE. Defualt is no source term
    p (numpy.ndarray): Numpy array of the parameters contained in the source term, given in the order they appear. Default is no parameters
    in_guess (numpy.ndarray): Numpy array of an initial guess for the solution if required. Default is no initial guess
    boundary_conds (list of two tuples): A list containing two tuples, where each tuple represents the boundary condition for one end of the domain
    tol (float): Tolerance for the scipy root function
    
    Returns:
    --------
    
    sol.x (numpy.ndarray): Solution of the ODE at the specified grid points
    """
    # defining the first eqaution
    if boundary_conds[0][0]=='D':
        f1_eq= lambda u,q_eval: np.array([(D*(u[1]-2*u[0]+boundary_conds[0][1])/(dx)**2)+P*((u[1]-boundary_conds[0][1])/(2*dx))+q_eval[0]])
    elif boundary_conds[0][0]=='N':
        f1_eq= lambda u,q_eval: np.array([D*(2*u[1]-2*u[0])/(dx)**2-(D*(2*boundary_conds[0][1])/dx)+P*(boundary_conds[0][1])+q_eval[0]])
    else:
        f1_eq= lambda u,q_eval: np.array([(-2*D(1-boundary_conds[0][2]*dx)*u[0]+2*u[1])/(dx)**2-(D*(2*boundary_conds[0][1])/dx)+P*(-boundary_conds[0][1]+boundary_conds[0][2]*u[0])+q_eval[0]])
    # defining the last equation
    if boundary_conds[1][0]=='D':
        f_end_eq= lambda u,q_eval: np.array([(D*(boundary_conds[1][1]-2*u[-1]+u[-2])/(dx)**2)+P*(boundary_conds[1][1]-u[-2])+q_eval[-1]])
    elif boundary_conds[1][0]=='N':
        f_end_eq= lambda u,q_eval: np.array([D*((-2*u[-1]+2*u[-2])/(dx)**2)+(D*(2*boundary_conds[1][1])/dx)+P*boundary_conds[1][1]+q_eval[-1]])
    else:
        f_end_eq= lambda u,q_eval: np.array([(D*(-2*(1+boundary_conds[1][2]*dx)*u[-1]+2*u[-2])/(dx)**2)+(D*(2*boundary_conds[1][1])/dx)+P*(boundary_conds[1][1]+boundary_conds[1][2]*u[-1])+q_eval[-1]])
    if in_guess is None:
        in_guess=np.zeros(len(x))
    # defining the function to find the root of
    def f(u):
        #evaluating q
        q_eval=q(x,u,p)
        # using the appropriate first and last equations
        f1=f1_eq(u,q_eval)
        f_inner=D*((u[2:]-2*u[1:-1]+u[0:-2])/(dx)**2)+P*((u[2:]-u[:-2])/(2*dx))+q_eval[1:-1]
        f_end=f_end_eq(u,q_eval)
        return np.concatenate([f1,f_inner,f_end])
    # finding the solution of the system of equations
    sol=spo.root(f,in_guess,tol=tol)
    return sol.x

def thomas_solve(c,b,a,d):
    """
    Description:
    ------------
    
    Solve a tridiagonal system of linear equations Ax = d using the Thomas algorithm.
    
    Args:
    -----
    c (numpy.ndarray): Upper diagonal elements of the tridiagonal matrix A
    b (numpy.ndarray): Main diagonal elements of the tridiagonal matrix A
    a (numpy.ndarray): Lower diagonal elements of the tridiagonal matrix A
    d (numpy.ndarray): Right-hand side vector

    Returns:
    --------
    
    x (numpy.ndarray): Solution vector x
    """
    # initialising intermediate arrays with the correct length
    n=len(d)
    c_dash=np.zeros(n-1,float)
    d_dash=np.zeros(n,float)
    # setting first values
    c_dash[0]=c[0]/b[0]
    d_dash[0]=d[0]/b[0]
    # performing the forward sweep
    for i in range(1,n-1):
        denominator = b[i]-a[i-1]*c_dash[i-1]
        c_dash[i] = c[i]/denominator
        d_dash[i] = (d[i]-a[i-1]*d_dash[i-1])/denominator
    # finding the last value 
    d_dash[-1]=(d[-1]-a[-1]*d_dash[-2])/(b[-1]-a[-1]*c_dash[-1])
    # initialising the solution vector
    x=np.zeros(n)
    # finding the last element
    x[-1]=d_dash[-1]
    # performing the backward sweep to find the solution vector
    for i in range(n-2,-1,-1):
        x[i]=d_dash[i]-c_dash[i]*x[i+1]
    return x
def pde_solve(a,b,dx,dt,D,t0,tfin,boundary_conds,in_cond,method='exp',solver='euler',in_cond_param=None,q=None,q_param=None,format='dense'):
    """
    Description:
    ------------

    Function which solves second order PDEs (Diffusion Equation) of the form ∂u/∂t = D * (∂^2u/∂x^2)^2 using explicit or implicit methods
    
    Args:
    -----
    a (float): x value of the start of the domain
    b (float): x value of the end of the domain
    dx (float): Distance between each x grid point
    dt (float): Distance between each t grid point
    D (float): Value of D in the PDE (coefficient of the second order derivative)
    t0 (float): Initial value of time
    tifn (float): Final value of time
    boundary_conds (list of two tuples): A list containing two tuples, where each tuple represents the boundary condition for one end of the domain.
                                         Each tuple should contain two elements: the type of boundary condition ('D', 'N', or 'R') and the corresponding
                                         boundary value. For Robin ('R') conditions, you may enter two values, where the first corresponds to the constant,
                                         and the second to the (negated) coefficient of u(x)
    in_cond (fun): Function in_cond(x,p), which returns a numpy array specifying the initial values for all grid points
    method (str): Method used to solve the pde. Options are 'exp' and 'imp', with default being 'exp'
    solver (str): Solver used. Options are 'thomas', 'numpy' and 'scipy' for 'imp', and 'euler' or 'rk4' for 'exp'. Default is 'exp'
    in_cond_param (numpy.ndarray): Array of the parameters for the initial conditions, given in the order they appear. Default is no parameters
    q (fun): Function q(t,x,u,p) which represents the source term in the PDE. Defualt is no source term
    q_param (numpy.ndarray): Numpy array of the parameters contained in the source term, given in the order they appear. Default is no parameters
    format (str): Format of matrices used. Default is 'dense'. Note that format='sparse' is only applicable for method='imp' and solver='scipy' 
    
    Returns:
    --------

    t (numpy.ndarray): Vector of the times that the PDE is solved at
    x (numpy.ndarray): Vector of the x-values that the PDE is solved at
    u (numpy.ndarray): Matrix of the solution at the corresponding t and x values
    """
    if method=='imp':
        # solving with implicit method
        t,x,u=imp_euler(a,b,dx,dt,D,t0,tfin,boundary_conds,in_cond,in_cond_param,q,q_param,solver,format)
    elif method=='exp':
        # solving with explicit method
        t,x,u=meth_of_lines(a,b,dx,dt,D,t0,tfin,boundary_conds,in_cond,in_cond_param,q,q_param,solver,format)
    else:
        raise ValueError(f"{method} is not a valid input for 'method'. Please input 'exp' or 'imp'")
    return t,x,u

def set_q_eval(x,q,q_param):
    """
    Description:
    ------------

    Function which defines the q_eval if none is provided, and evaluates provided q_eval functions at the grid points and parameter values
    
    Args:
    -----
    
    x(numpy.ndarray): Array containing the grid points
    q (fun): Function q(t,x,u,p) which represents the source term in the PDE
    q_param (numpy.ndarray): Numpy array of the parameters contained in the source term, given in the order they appear
    
    Returns:
    --------

    q_eval (fun): Users original souorce term, or if none is provided the default source term (vector of zeros)
    """
    # setting the source term to be zero if none was provided
    if q is None:
        def q_eval(t,u):
            return np.zeros(len(x))
    else:
        # evaluating user source term at grid points and parameters
        q_eval = lambda t,u: q(t,x,u,q_param)
    return q_eval

def meth_of_lines(a,b,dx,dt,D,t0,tfin,boundary_conds,in_cond,in_cond_param,q,q_param,solver,format):
    """
    Description:
    ------------

    Function which solves second order PDEs (Diffusion Equation) of the form ∂u/∂t = D * (∂^2u/∂x^2)^2 using the method of lines  
    
    Args:
    -----
    
    a (float): x value of the start of the domain
    b (float): x value of the end of the domain
    dx (float): Distance between each x grid point
    dt (float): Distance between each t grid point
    D (float): Value of D in the PDE (coefficient of the second order derivative)
    t0 (float): Initial value of time
    tifn (float): Final value of time
    boundary_conds (list of two tuples): A list containing two tuples, where each tuple represents the boundary condition for one end of the domain.
                                         Each tuple should contain two elements: the type of boundary condition ('D', 'N', or 'R') and the corresponding
                                         boundary value. For Robin ('R') conditions, you may enter two values, where the first corresponds to the constant,
                                         and the second to the (negated) coefficient of u(x)    
    in_cond (fun): Function in_cond(x,p), which returns a numpy array specifying the initial values for all grid points
    in_cond_param (numpy.ndarray): Array of the parameters for the initial conditions, given in the order they appear
    q (fun): Function q(t,x,u,p) which represents the source term in the PDE
    q_param (numpy.ndarray): Numpy array of the parameters contained in the source term, given in the order they appear
    solver (str): Solver used. Options are 'euler' and 'rk4'
    format (str): Format of matrices used. Note that format='sparse' is only applicable for method='imp' and solver='scipy'
    
    Returns:
    --------
    
    t (numpy.ndarray): Vector of the times that the PDE is solved at
    x (numpy.ndarray): Vector of the x-values that the PDE is solved at
    u (numpy.ndarray): Matrix of the solution at the corresponding t and x values
    """    
    # finding the value of C
    C=((dx)**2)/(2*D)
    if dt>C:
        # warning the user if their value dt violates the stability conditions
        if solver=='euler':
            raise ValueError(f'The chosen value of dt violates the Explicit Euler stability condition. Please enter a dt value of {C} or below.')
        elif solver=='rk4':
            warnings.warn(f"Choice of dt value potentially violates Explicit rk4 stability condition. If solver fails, consider reducing the size of dt.",stacklevel=2)
    # finding the number of grid steps to take
    N=int(abs((b-a)/dx))   
    # finding and reforming the matrices 
    diags,b_vec,_,x=grid(N,a,b,boundary_conds,dx,format)
    A,_=reform_matrices(diags,format)
    # setting q_eval
    q_eval=set_q_eval(x,q,q_param)
    # finding the number of timesteps
    N_t=int(np.ceil((tfin-t0)/dt))
    # setting the t vector
    t=np.linspace(t0,tfin,N_t+1)
    # setting the initial value of u
    u0=in_cond(x,in_cond_param)
    # defining the function for input to solve_to
    if callable(b_vec):
        # treating the b vector as a function if time dependent conditions
        def f(t,u,p):
            return (D/(dx)**2*(p[0]@u+p[1](t)))+q_eval(t,u)
    else:
        def f(t,u,p):
            return (D/(dx)**2*(p[0]@u+p[1]))+q_eval(t,u)
    # defining the matrix and b vector as parameters to pass them to solve to - allows b_vec to be a function of time
    p=[A,b_vec]
    # solving using scipy solve_ivp
    if solver=='scipy':
        tspan=(t0,tfin)
        sol=spi.solve_ivp(f,tspan,u0,args=(p,)) 
        t=sol.t
        u=sol.y
    else:
        # solve using solve_to
        t,u=solve_to(f,p,t0,u0,tfin,dt,solver)
    # appending value at 'a' to the solution if LH boundary cond is Dirichlet
    if boundary_conds[0][0]=='D':
        u=np.vstack([boundary_conds[0][1]*np.ones([len(t)]),u])
    # appending value at 'b' to the solution if RH boundary cond is Dirichlet
    if boundary_conds[1][0]=='D':
        u=np.vstack([u,boundary_conds[1][1]*np.ones([len(t)])])
    # reforming the complete x vector
    x=np.linspace(a,b,N+1)
    return t,x,u

def imp_euler(a,b,dx,dt,D,t0,tfin,boundary_conds,in_cond,in_cond_param,q,q_param,solver,format):
    """
    Description:
    ------------

    Function which solves second order PDEs (Diffusion Equation) of the form ∂u/∂t = D * (∂^2u/∂x^2)^2 using implicit methods (and IMEX methods for problems with source terms)
    
    Args:
    -----
    
    a (float): x value of the start of the domain
    b (float): x value of the end of the domain
    dx (float): Distance between each x grid point
    dt (float): Distance between each t grid point
    D (float): Value of D in the PDE (coefficient of the second order derivative)
    t0 (float): Initial value of time
    tifn (float): Final value of time
    boundary_conds (list of two tuples): A list containing two tuples, where each tuple represents the boundary condition for one end of the domain.
                                         Each tuple should contain two elements: the type of boundary condition ('D', 'N', or 'R') and the corresponding
                                         boundary value. For Robin ('R') conditions, you may enter two values, where the first corresponds to the constant,
                                         and the second to the (negated) coefficient of u(x)    
    in_cond (fun): Function in_cond(x,p), which returns a numpy array specifying the initial values for all grid points
    in_cond_param (numpy.ndarray): Array of the parameters for the initial conditions, given in the order they appear
    q (fun): Function q(t,x,u,p) which represents the source term in the PDE
    q_param (numpy.ndarray): Numpy array of the parameters contained in the source term, given in the order they appear
    solver (str): Solver used. Options are 'euler' and 'rk4'
    format (str): Format of matrices used. Note that format='sparse' is only applicable for method='imp' and solver='scipy'
    
    Returns:
    --------
    
    t (numpy.ndarray): Vector of the times that the PDE is solved at
    x (numpy.ndarray): Vector of the x-values that the PDE is solved at
    u (numpy.ndarray): Matrix of the solution at the corresponding t and x values
    """ 
    # finding the number of grid steps
    N=int(abs((b-a)/dx))   
    # findnig the matrices
    diags,b_vec,_,x=grid(N,a,b,boundary_conds,dx,format)
    # finding the number of timesteps
    N_t=int(np.ceil(((tfin-t0)/dt)))
    # setting the t vector
    t=np.linspace(t0,tfin,N_t+1)
    # finding the first solution
    sol_p=in_cond(x,in_cond_param)
    # setting the solution array
    u=np.zeros([len(x),N_t+1])
    # setting the initial solution
    u[:,0]=sol_p
    # finding the value of C
    C=(D*dt)/(dx**2)
    # setting q_eval
    q_eval=set_q_eval(x,q,q_param)
    if solver=='thomas':
        # setting diagonals for thomas algorithm
        A_sup_diag,A_main_diag,A_sub_diag,_,_,_=diags
        main_diag=np.ones(len(x))-C*A_main_diag
        sup_diag=-C*A_sup_diag
        sub_diag=-C*A_sub_diag
    else:
        # reforming matrix from diags
        A,_=reform_matrices(diags,format)
        if solver=='numpy':
            # setting up dense identity matrix
            identity_mat=np.eye(len(x))
            # setting solver to numpy sovler
            solve=np.linalg.solve
        elif solver=='scipy':
            # setting up sparse identity matrix
            identity_mat=spp.eye(len(x))
            # setting solver to scipy sparse solver
            solve=spp.linalg.spsolve
        else:
            raise ValueError(f"{solver} is not a valid input for 'solver'. Please input 'numpy', 'scipy' or 'thomas'")
    for idx in range(1,N_t+1):
        if solver=='thomas':
            # solving using thomas algorithm
            sol_n=thomas_solve(sup_diag,main_diag,sub_diag,sol_p+C*b_vec+dt*q_eval(t,sol_p))
        else:
            # solving using previously specified solver
            sol_n=solve(identity_mat-C*A,sol_p+C*b_vec+dt*q_eval(t,sol_p))
        # storing solution in solution array
        u[:,idx]=sol_n
        # setting calculated solution to previous solution for next iteration
        sol_p=sol_n
    # appending value at 'a' to the solution if LH boundary cond is Dirichlet
    if boundary_conds[0][0]=='D':
        u=np.vstack([boundary_conds[0][1]*np.ones([N_t+1]),u])
    # appending value at 'b' to the solution if RH boundary cond is Dirichlet
    if boundary_conds[1][0]=='D':
        u=np.vstack([u,boundary_conds[1][1]*np.ones([N_t+1])])
    # reforming the complete x vector
    x=np.linspace(a,b,N+1)
    return t,x,u
        

    

