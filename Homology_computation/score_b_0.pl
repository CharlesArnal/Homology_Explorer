# @ARGV should be local_path, signs_input_file_name, triangs_input_file_name, points_input_file_name,
#                    0                  1                         2                       3        
#  relevant_points_input_file_name, output_file_name, save_homologies, homologies_output_file
#                4                          5                 6                7


# Two cases : either there are as many signs as there triangulations (associated to each other),
# or there is a single triangulation and many signs


use application "tropical";

$Verbose::credits = 0;
$Polymake::User::Verbose::cpp =1;


my @signs =();
open(INPUT, "<", "$ARGV[0]/$ARGV[1]");
while(<INPUT>)
{
    if($_ ne "\n")
    {push(@signs, $_) ;}
}
close(INPUT);

my @triangs =();
open(INPUT, "<", "$ARGV[0]/$ARGV[2]");
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

open(INPUT, "<", "$ARGV[0]/$ARGV[3]");
my $points = new Matrix<Rational>(<INPUT>);
close(INPUT);

my @relevant_indices;
open(INPUT, "<", "$ARGV[0]/$ARGV[4]");
while(<INPUT>)
{
    my $input = $_;
    # replace the remaining commas by blank spaces : {0,1,2}{0,1,3} -> {0 1 2}{0 1 3}
    $input =~ s/,/ /ig;
    push(@relevant_indices, new Set<Int>($input));
}
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


my @score_array=();
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
    if($ARGV[6] eq "True")
    {
        push(@homologies_array,$h1->PATCHWORK(SIGNS=>$signs[$_])->BETTI_NUMBERS_Z2);
        push(@score_array,$homologies_array[-1]->[0]);
        $h1->remove("PATCHWORK");
    }
    else
    {
        push(@score_array,$h1->PATCHWORK(SIGNS=>$signs[$_])->BETTI_NUMBERS_Z2->[0]);
        $h1->remove("PATCHWORK");
    }
}


open(my $f, ">", "$ARGV[0]/$ARGV[5]");
print $f join("\n",@score_array);
#print $f "$_\n" for @score_array ;
close $f;

if($ARGV[6] eq "True")
{
    open(my $f, ">", "$ARGV[0]/$ARGV[7]");
    print $f join("\n",@homologies_array);
    #print $f "$_\n" for @score_array ;
    close $f;
}