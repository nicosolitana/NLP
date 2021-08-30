import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import java.util.*;
public class FPOST{
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in); 
        MaxentTagger tagger =  new MaxentTagger("filipino-left5words-owlqn2-distsim-pref6-inf2.tagger");
        while(true){
            System.out.println("Magbigay ng pangungusap (i-type ang \"exit\" para matapos ang program): ");
            String pangungusap = scan.nextLine();
            if(pangungusap.equals("exit")) break;
            else {
                String tagged = tagger.tagString(pangungusap);
                System.out.println(tagged);
            }
        }
    }
}