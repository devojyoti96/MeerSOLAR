MeerSOLAR Flowchart
=====================

The pipeline follows several steps. By default, all steps are executed. If the user desires, they can selectively disable any step. However, the pipeline will only function correctly if its internal logic is maintained.

For example, if the user disables splitting the target scans and no target directory exists beforehand, imaging will not be performed. However, if self-calibration is disabled, the pipeline will skip that step and proceed with applying basic calibrations and generating final images.

.. admonition:: Recommendation
   :class: tip

   It is recommended to go through the flowchart of the pipeline and understand it before modifying the pipeline keys.

.. admonition:: Click here to see the MeerSOLAR pipeline flowchart
   :class: dropdown

   .. mermaid::

      graph TD
          Start([Start])
          Decision1{HPC?}
          Process1[Fluxcal with<br>noise-diode]
          Process2[Target splitting<br>in parallel]
          Decision2{Do basic<br>calibration?}
          Process3[Make multi-ms using<br>calibrator scans]
          Process4[Perform flagging<br>on calibrators]
          Process4a[Simulate<br>calibrator<br>visibilities]
          Process5[Perform basic<br>calibration]
          Decision2a{Calibration<br>table<br>present?}
          Decision3{Do<br>self<br>calibration?}
          Process6[Apply basic<br>calibrations]
          Decision1a{HPC?}
          Process7[Target splitting<br>in parallel]
          Process8[Perform<br>self-calibration]
          Process9([Stop with<br>basic calibrated<br>visibilities])
          Decision4{Self<br>calibration<br>successful?}
          Process10[Apply<br>self-calibration]
          Process11([Stop with<br>basic calibrated<br>visibilities])
          Process12[Split raw<br>data of<br>target scans<br>for imaging]
          Process13[Apply<br>basic calibrations]
          Process14[Apply<br>self calibrations]
          Process15[Perform imaging]
          Process16([Finished with<br>final imaging<br>products])
          Stop([Pipeline end])

          Start --> Decision1
          Decision1 -->|Y/N| Process1
          Process1 --> Decision2
          Decision1 -->|Y| Process2
          Decision2 -->|Y| Process3
          Process3 --> Process4 --> Process4a --> Process5
          Process5 --> Decision2a
          Decision2 -->|N| Decision2a
          Decision2a -->|Y| Process6
          Decision2a -->|N| Stop
          Process6 --> Decision3
          Decision3 -->|Y| Decision1a
          Decision1a -->|Y| Process8
          Decision1a -->|N| Process7 --> Process8
          Process8 --> Decision4
          Decision3 -->|N| Process9
          Decision4 -->|Y| Process10 --> Process12 --> Process13 --> Process14 --> Process15 --> Process16
          Decision4 -->|N| Process11

