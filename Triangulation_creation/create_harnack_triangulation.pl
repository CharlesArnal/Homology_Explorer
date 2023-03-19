# $ARGV is degree       full_path_to_triang_file    full_path_to_points_file
#           0                   1                           2
use application "tropical";

my $degree = int($ARGV[0]);


my $harnack = harnack_curve($degree);

open(my $f, ">", "$ARGV[2]");
print $f $harnack->DUAL_SUBDIVISION->POLYHEDRAL_COMPLEX->VERTICES;
close $f;

open(my $f, ">", "$ARGV[1]");
my $triang = $harnack->DUAL_SUBDIVISION->MAXIMAL_CELLS;
$triang =~ s/\n//ig;
$triang = "{" . $triang . "}";
print $f $triang;
close $f;


#$C=new Hypersurface<Min>(MONOMIALS=>[ [2,0,0],[1,1,0],[0,2,0],[1,0,1],[0,1,1],[0,0,2] ], COEFFICIENTS=>[6,5,6,5,5,7]);
