MeerSOLAR Flowchart
=====================
The pipeline follows several steps. By default, all steps are done. If user want, they can switch off any step. However, it will work only if the pipeline logic is still maintained.

For example, if user switch off spliting the target scans, there will not be any target directory, if not present alreayd. In that case, no imaging will be performed. But, if user switched off self-calibration, pipeline will not perform self-calibration. Pipeline now only apply basic calibration and make the final images. 

.. admonition:: Recommendation
   :class: tip
    
   It is recommended to go through the flowchart of the pipeline and understand before playing with the pipeline keys.

.. admonition:: Click here to see the MeerSOLAR pipeline flowchart
   :class: dropdown
   
   .. graphviz::

      digraph G {
           rankdir=TB;  // Top to Bottom
           
           Start [label="Start", shape=ellipse];
           Decision1 [label="HPC?", shape=diamond];
           Process1 [label="Fluxcal with\nnoise-diode", shape=box];
           Process2 [label="Target spliting\nin parallel", shape=box];
           Decision2 [label="Do basic\ncalibration?", shape=diamond];
           Process3 [label="Make multi-ms using\ncalibrator scans", shape=box];
           Process4 [label="Perform flagging\non calibrators", shape=box];
           Process4a [label="Simulate\ncalibrator\nvisibilities", shape=box];
           Process5 [label="Perform basic\ncalibration", shape=box];
           Decision2a [label="Calibration\ntable\npresent?", shape=diamond];
           Decision3 [label="Do\nself\ncalibration?", shape=diamond];
           Process6 [label="Apply basic\ncalibrations", shape=box];
           Decision1a [label="HPC?", shape=diamond];
           Process7 [label="Target spliting\nin parallel", shape=box];
           Process8 [label="Perform\nself-calibration", shape=box];
           Process9 [label="Stop with\nbasic calibrated\nvisibilities", shape=ellipse];
           Decision4 [label="Self\ncalibration\nsuccessful?", shape=diamond];
           Process10 [label="Apply\nself-calibration", shape=box];
           Process11 [label="Stop with\nbasic calibrated\nvisibilities", shape=ellipse];
           Process12 [label="Split raw\ndata of\ntarget scans\nfor imaging", shape=box];
           Process13 [label="Apply\nbasic calibrations", shape=box];
           Process14 [label="Apply\nself calibrations", shape=box];
           Process15 [label="Perform imaging", shape=box];
           Process16 [label="Finished with\nfinal imaging\nproducts", shape=ellipse];
           Stop [label="Pipeline end", shape=ellipse];
          
          
           Start -> Decision1;
           Decision1 -> Process1 [label=" Y/N", tailport=s, headport=n, rank=same];
           Process1 -> Decision2 [tailport=s, headport=n, rank=same];
           Decision1-> Process2 [label=" Y", tailport=e, headport=n, rank=same];
           Decision2 -> Process3 [label=" Y", tailport=w, headport=n, rank=same];
           Process3 -> Process4 -> Process4a-> Process5 [tailport=s, headport=n, rank=same]; 
           Process5 -> Decision2a [tailport=s, headport=w, rank=same]; 
           Decision2 -> Decision2a [label=" N", tailport=s, headport=n, rank=same];
           Decision2a -> Process6 [tailport=s, headport=n, rank=same];
           Decision2a -> Stop [tailport=e, headport=n, rank=same];
           Process6 -> Decision3 [tailport=s, headport=n, rank=same];
           Decision3 -> Decision1a [label=" Y", tailport=w, headport=n, rank=same]; 
           Decision1a -> Process8 [label=" Y", tailport=e, headport=n, rank=same]; 
           Decision1a -> Process7 [label=" N", tailport=s, headport=n, rank=same]; 
           Process7 -> Process8 [tailport=s, headport=w, rank=same];
           Process8 -> Decision4 [tailport=s, headport=n, rank=same];
           Decision3 -> Process9 [label=" N", tailport=e, headport=n, rank=same];
           Decision4 -> Process10 [label=" Y", tailport=s, headport=n, rank=same]
           Decision4 -> Process11 [label=" N", tailport=e, headport=n, rank=same];
           Process10 -> Process12 -> Process13 -> Process14 -> Process15 -> Process16 [tailport=s, headport=n, rank=same];            
       }
   
   
   
   
   
   
