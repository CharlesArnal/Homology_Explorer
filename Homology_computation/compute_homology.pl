# @ARGV should be signs_input_file_name, triangs_input_file_name, points_input_file_name,
#                           0                  1                         2                
#  output_file_name
#          3   

# Takes full paths

use application "tropical";

$Verbose::credits = 0;
$Polymake::User::Verbose::cpp =1;


my @signs =();
open(INPUT, "<", "$ARGV[0]");
while(<INPUT>)
{
    if($_ ne "\n")
    {push(@signs, $_) ;}
}
close(INPUT);

my @triangs =();
open(INPUT, "<", "$ARGV[1]");
while(<INPUT>)
{
    if($_ ne "\n")
    {
        my $input = $_;
        # remove the line breaks
        $input =~ s/\n//ig;
        # remove the first and last {} : {{0,1,2},{0,1,3}} -> {0,1,2},{0,1,3}
        $input = substr($input,1,length($input)-2);
        # remove the commas between simplices : {0,1,2},{0,1,3} -> {0,1,2}{0,1,3}
        $input =~ s/},{/}{/ig;
        # replace the remaining commas by blank spaces : {0,1,2}{0,1,3} -> {0 1 2}{0 1 3}
        $input =~ s/,/ /ig;
        push(@triangs, new IncidenceMatrix<NonSymmetric>($input)) ;
    }
}
close(INPUT);

open(INPUT, "<", "$ARGV[2]");
my $points = new Matrix<Rational>(<INPUT>);
close(INPUT);




my $dual_sub;
my $h1;

# If there is a single triangulation, we use it for all signs distributions
# Else, we use a different triangulation for each signs distribution
if(scalar @triangs ==1)
{
    $dual_sub = new fan::SubdivisionOfPoints(POINTS=>$points,MAXIMAL_CELLS=>$triangs[0]);
    $h1 = new Hypersurface<Min>(DUAL_SUBDIVISION=>$dual_sub);
}


my @homologies_array=();

my $n_signs = scalar @signs;
my @a = (0..$n_signs-1);
foreach(@a)
{

    if (scalar @triangs !=1)
    {
        $dual_sub = new fan::SubdivisionOfPoints(POINTS=>$points,MAXIMAL_CELLS=>$triangs[$_]);
        $h1 = new Hypersurface<Min>(DUAL_SUBDIVISION=>$dual_sub);
    }
    push(@homologies_array,$h1->PATCHWORK(SIGNS=>$signs[$_])->BETTI_NUMBERS_Z2);
}


open(my $f, ">", "$ARGV[3]");
print $f join("\n",@homologies_array);
close $f;
