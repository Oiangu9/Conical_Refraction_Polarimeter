import numpy as np
from time import time

class Ad_Hoc_Optimizer:
    def __init__(self, min_in_range, max_in_range, initial_guess_delta, evaluate_cost, fib_prec=None):
        self.a=min_in_range
        self.b=max_in_range
        self.initial_guess_delta = initial_guess_delta
        self.evaluate_cost=evaluate_cost
        if fib_prec is None:
            self.F_n=None
        else:
            self.compute_fib_iteration_for_prec(fib_prec)

    def _round_to_sig(self, x_to_round, reference=None, sig=2):
        reference = x_to_round if reference is None else reference
        return round(x_to_round, sig-int(np.floor(np.log10(abs(reference))))-1)

    def brute_force_search(self, steps, zoom_ratios, image, args_for_cost):
        """
        Arguments
        --------
        - steps (list): A list of the different feasible point steps to take in each of the sweeps.
            Expected N, where N is the number of sweeps that will be performed. The first one is
            expected to be the coarsest grain and they should be ordered from big to smallest.
            The last step in the list will define the precision of the found minimum. The point
            steps are expected to be in (0,b-a).

        - zoom_ratios (list): A list of the interval reductions that will be held after each sweep
            around the current best candidate for the minimum. There should be N-1 elements and
            they should be numbers in (0,1]. It should have an extra last element to account for
            the last iteration.
        Returns
        -------
        time, computed_points, optimal, optimum, precision
        """
        if not isinstance(args_for_cost, (list, tuple)):
            args_for_cost = (args_for_cost,)
        optimals={}
        optimums={}
        times={}
        precisions={}
        computed_points={}
        a, b=self.a, self.b
        # Execute all the stages
        for stage, step in enumerate(steps):
            t=time()
            feasible_points = np.arange(start=a, stop=b, step=step, dtype=np.float64)
            costs = [self.evaluate_cost(image, point, *args_for_cost) for point in feasible_points]
            x_min = feasible_points[np.argmin(costs)] # pero seria maximo
            a=x_min-(b-a)*zoom_ratios[stage]/2.0
            b=x_min+(b-a)*zoom_ratios[stage]/2.0
            t = time()-t
            times[f"Stage_{stage}"]=self._round_to_sig(t)
            computed_points[f"Stage_{stage}"]=np.stack((feasible_points, costs)).transpose()
            optimals[f"Stage_{stage}"] = self._round_to_sig(x_min, step)
            optimums[f"Stage_{stage}"] = np.min(costs)
            precisions[f"Stage_{stage}"]=step
        return times, computed_points, optimals, optimums, precisions

    def compute_fib_iteration_for_prec(self, precision):
        # we compute the necessary fibonacci iteration to achieve this precision
        F_Nplus1 = (self.b-self.a)/(2.0*precision)+1
        self.F_n=[1.0,1.0,2.0]
        # compute fibonacci series till there
        while self.F_n[-1]<=F_Nplus1:
            self.F_n.append(self.F_n[-1]+self.F_n[-2])

    def fibonacci_ratio_search(self, precision, maximum_points, cost_tol, image, args_for_cost, initial_guess=None):
        """
        Arguments
        --------
        - precision (float): Half the length of the interval achieved in the last step. It will be
            the absolute error to which we compute the minimum of the funtion. Note however that
            the precision will have a minimum depending on the image quality and the minimum
            cost evaluation arithmetics.
            Noise or discretization can induce plateaus in the minimum.

            Therefore, if at some point the three points have the same cost function the algorithm
            will stop: the cost function has arrived to the plateau. In that case the precision
            will be outputed accordingly.

        - maximum_points (int): Maximum number of points to use in the minimum search. It is also
            the number of times to make an interval reduction.

        - cost_tol (float): Maximum relative difference between the cost function active points
            tolerated before convergence assumption.

        Returns
        -------
        time, computed_points, optimal, optimum, precision
        """
        if not isinstance(args_for_cost, (list, tuple)):
            args_for_cost = (args_for_cost,)
        if self.F_n is None:
            self.compute_fib_iteration_for_prec(precision)

        t=time()
        # prepare the first point triad just like in quadratic fit search
        active_points = self.initialize_correct_point_quad(image, args_for_cost, initial_guess)
        # for plotting
        computed_points = active_points.transpose().tolist() # list of all the pairs of (xj,f(xj))

        for it in range(len(self.F_n)-1):
            rho = 1-self.F_n[-1-it-1]/self.F_n[-1-it] # interval reduction factor then is (1-rho)
            min_triad = np.argmin(active_points[1]) # 1 or 2 if first 3 or last 3
            if min_triad==1: # we preserve the first 3 points of the quad
                if rho<1.0/3:
                    active_points[0,-1]=active_points[0,2]-rho*(active_points[0,2]-active_points[0,0])
                else:
                    active_points[0,-1]=active_points[0,0]+rho*(active_points[0,2]-active_points[0,0])
                active_points[1,-1] = self.evaluate_cost(image, active_points[0,-1], *args_for_cost)
                # save new point for plotting
                computed_points.append(active_points[:,0])
            else: # if 2, we preserve last 3 points
                if rho>1.0/3:
                    active_points[0,0]=active_points[0,3]-rho*(active_points[0,3]-active_points[0,1])
                else:
                    active_points[0,0]=active_points[0,1]+rho*(active_points[0,3]-active_points[0,1])
                active_points[1,0] = self.evaluate_cost(image, active_points[0,0], *args_for_cost)
                # save new point for plotting
                computed_points.append(active_points[:,0])

            # order the four pairs of points by their angle
            active_points = active_points[:, np.argsort(active_points[0])]

            if np.abs(active_points[0,-1]-active_points[0,0]) < 2*precision or np.allclose(active_points[1,:], active_points[1,0], rtol=cost_tol) or it==maximum_points:
                break
        t = time()-t
        # save all the data
        min_point = np.argmin(active_points[1]) # index of minimum f(xj) from the four active points
        ach_precision = self._round_to_sig((active_points[0, min_point+1]-active_points[0, min_point-1])/2.0)

        return (self._round_to_sig(t), np.array(computed_points),
            self._round_to_sig(active_points[0,min_point], ach_precision),
            active_points[1,min_point], ach_precision)

    def initialize_correct_point_quad(self, image, args_for_cost, initial_guess=None):
        """
        We initialize a point quad where the minimum of the cost function is for sure not in the
        boundaries of the quad.
        This is necessary for the quadratic fit search, and at least convenient for the fibonacci
        search.

        Returns an array [2,4] with the zero-th row having the ordered feasible points
        and the first row their cost function values, such that the minimum cost of the four
        pairs of points is in position 1 or 2.
        """
        # Initialize the active points to consider in the first iteration
        if initial_guess==None:
            active_xs = np.array([self.a,
                                0.5*(self.b+self.a-self.initial_guess_delta),
                                0.5*(self.b+self.a+self.initial_guess_delta),
                                self.b], dtype=np.float64)
        else: # we have already a candidate for the minimum
            active_xs = np.array([self.a,
                                initial_guess-self.initial_guess_delta,
                                initial_guess+self.initial_guess_delta,
                                self.b], dtype=np.float64)

        # Evaluate cost function for each angle
        #print("active_xs",active_xs, "costos", [ self.evaluate_cost(image, angle, *args_for_cost) for angle in active_xs])
        active_points = np.stack((active_xs, [ self.evaluate_cost(image, angle, *args_for_cost) for angle in active_xs])) # [2 (xj,f(xj)),4]
        # if the minium is in the boundary of the interval, make it not be the boundary. we subtract the initial_guess_delta to break the symmetries
        if np.argmin(active_points[1])==0:
            active_points[0, 3] -= 3*(self.b-self.a)/2+self.initial_guess_delta
            active_points[1,3] = self.evaluate_cost(image, active_points[0, 3], *args_for_cost)
        elif np.argmin(active_points[1])==3:
            active_points[0, 0] += 3*(self.b-self.a)/2+self.initial_guess_delta
            active_points[1,0] = self.evaluate_cost(image, active_points[0,0], *args_for_cost)

        # order the four pairs of points by their support position
        return active_points[:, np.argsort(active_points[0])]

    def quadratic_fit_search(self, precision, max_iterations, cost_tol, image, args_for_cost, initial_guess=None):
        """
        Arguments
        --------
        - precision (float): Half the length of the interval achieved in the last step. It will be
            the absolute error to which we compute the minimum of the function. Note however that
            the precision will have a minimum depending on the image quality and the minimum
            cost eval. arithmetics. Noise or discretization can induce plateaus in the minimum.
            Therefore, if at some point the three points have the same cost function the algorithm
            will stop: the cost function has arrived to the plateau. In that case the precision
            will be outputed accordingly.

        - max_iterations (int): Number of maximum iterations of quadratic function fit and
            minimization to tolerate.

        - cost_tol (float): Maximum relative difference between the cost function active points
            tolerated before convergence assumption.

        Returns
        -------
        time, computed_points, optimal, optimum, precision

        """
        if not isinstance(args_for_cost, (list, tuple)):
            args_for_cost = (args_for_cost,)
        t=time()
        it=0
        active_points = self.initialize_correct_point_quad( image, args_for_cost, initial_guess)

        computed_points = active_points.transpose().tolist() # list of all the pairs of (xj,f(xj))

        while( np.abs(active_points[0,-1]-active_points[0,0]) >= 2*precision and not (np.allclose(active_points[1,0], active_points[1,1], rtol=cost_tol) or np.allclose(active_points[1,2], active_points[1,1], rtol=cost_tol)) and it<=max_iterations):
            # Choose new triad of angles
            min_point = np.argmin(active_points[1]) # index of minimum f(xj) from the four active points
            # using the fact that the minimum of the four points will never be in the boundary
            active_points[:, :3] = active_points[:, (min_point-1):(min_point+2)]
            # compute the interpolation polynomial parameters and the minimum
            x_min = 0.5*( active_points[0,0]+active_points[0,1] + (active_points[0,0]-active_points[0,2])*(active_points[0,1]-active_points[0,2])/( ( active_points[0,0]*(active_points[1,2]-active_points[1,1])+active_points[0,1]*(active_points[1,0]-active_points[1,2]) )/(active_points[1,1]-active_points[1,0]) + active_points[0,2] ) )
            active_points[0,3] = x_min
            active_points[1,3] = self.evaluate_cost(image, x_min, *args_for_cost)

            # save new point for plotting
            computed_points.append(active_points[:,3])

            # order the four pairs of points by their angle
            active_points = active_points[:, np.argsort(active_points[0])]
            # increment iterations
            it+=1

        t = time()-t
        # save al the data
        min_point = np.argmin(active_points[1]) # index of minimum f(xj) from the four active points
        ach_precision = self._round_to_sig((active_points[0, min_point+1]-active_points[0, min_point-1])/2.0)
        return (self._round_to_sig(t), np.array(computed_points),
            self._round_to_sig(active_points[0,min_point], ach_precision),
            active_points[1,min_point], ach_precision)
