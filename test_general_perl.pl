# @ARGV should be local_path, signs_input_file_name, triangs_input_file_name, points_input_file_name,
#                    0                  1                         2                       3        
#  relevant_points_input_file_name, output_file_name, save_homologies, homologies_output_file
#                4                          5                 6                7


# Two cases : either there are as many signs as there triangulations (associated to each other),
# or there is a single triangulation and many signs


use List::Util qw(sum);
use application "tropical";

$Verbose::credits = 0;
$Polymake::User::Verbose::cpp =1;


my $v1 = 1;
my $v2 = 3;


print $v1 + 0.2*$v2;