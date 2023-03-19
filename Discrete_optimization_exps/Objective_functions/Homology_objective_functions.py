
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

from Current_Point import Current_Point

current_point = Current_Point(dim, degree,  local_path, current_point_folder)

# TODO : create current point, then :
# obj_function_for_signs_optimizer = create_objective_function_for_signs_optimization(self.list_of_homologies_file, self.temp_files_folder, polymake_scoring_script)
# return obj_function_for_signs_optimizer(current_point, solutions)
# think of what to put here and what to put in get_parameters 
# a priori, 3 parameters : degree, dim and actual objective function