///
/// This macro has been written for the Higgs to tau+tau- Kaggle Data Challenge
/// for use with the ATLAS-UK machine learning session.
///
///     root -l ./Htautau.C
///
/// The output file "TMVA.root" can be analysed with the use of dedicated
/// macros (simply say: root -l <macro.C>), which can be conveniently
/// 
/// Launch the GUI via the command (in a ROOT session):
///
///     TMVA::TMVAGui("TMVA.root")
///
/// Data for this example are available from the Kaggle website, the reduced
/// files used for the tutorial can be downloaded from the URL given in the 
/// corresponding slide deck at:
///
///     https://indico.cern.ch/event/759980/contributions/3259908/
///
/// Adrian Bevan (adapted from the TMVAClassification.C tutorial example)
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Tools.h"
#include "TMVA/TMVAGui.h"

int     Htautau( TString myMethodList="" );
int     main( int argc, char** argv );
TTree * readData(TString fname, TString treename);


int Htautau( TString myMethodList )
{
   // The explicit loading of the shared libTMVA is done in TMVAlogon.C, defined in .rootrc
   // if you use your private .rootrc, or run from a different directory, please copy the
   // corresponding lines from .rootrc
   //
   // Note the myMethodList argument functionality has been removed to focus on BDT training
   // with, and simplify, this example.

   //---------------------------------------------------------------
   // This loads the library
   TMVA::Tools::Instance();

   // Default MVA methods to be trained + tested
   std::map<std::string,int> Use;

   // Boosted Decision Trees - you can add additional methods to train other types of model
   // see the TMVAClassification.C example in $ROOTSYS/tutorials/tmva for details.
   Use["BDT"]             = 1; // uses Adaptive Boost

   std::cout << std::endl;
   std::cout << "==> Start Htautau" << std::endl;
   std::cout << "==>   This example is set to use the BDT with AdaBoost.M1" << std::endl;
   std::cout << "==>   For this tutorial we are going to train and optimise a model" << std::endl;


   // Select methods (don't look at this code - not of interest)
   if (myMethodList != "") {
      for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) it->second = 0;

      std::vector<TString> mlist = TMVA::gTools().SplitString( myMethodList, ',' );
      for (UInt_t i=0; i<mlist.size(); i++) {
         std::string regMethod(mlist[i]);

         if (Use.find(regMethod) == Use.end()) {
            std::cout << "Method \"" << regMethod << "\" not known in TMVA under this name. Choose among the following:" << std::endl;
            for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) std::cout << it->first << " ";
            std::cout << std::endl;
            return 1;
         }
         Use[regMethod] = 1;
      }
   }

   // --------------------------------------------------------------------------------------------------

   // Here the preparation phase begins

   // Read training and test data
   // (it is also possible to use ASCII format as input -> see TMVA Users Guide)
   TString sfname = "../data/train_sml_sig.csv";
   TString bfname = "../data/train_sml_bg.csv";

   // Register the training and test trees
   TTree *signalTree = readData(sfname, "signal");
   TTree *bgTree     = readData(bfname, "background");
   signalTree->Print("v");
   bgTree->Print("v");

   // Create a ROOT output file where TMVA will store ntuples, histograms, etc.
   TString outfileName( "TMVA.root" );
   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );

   // Create the factory object. Later you can choose the methods
   // whose performance you'd like to investigate. The factory is
   // the only TMVA object you have to interact with
   //
   // The first argument is the base of the name of all the
   // weightfiles in the directory weight/
   //
   // The second argument is the output file for the training results
   // All TMVA output can be suppressed by removing the "!" (not) in
   // front of the "Silent" argument in the option string
   TMVA::Factory *factory = new TMVA::Factory( "TMVAClassification", outputFile,
                                               "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification" );

   TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset");
   // If you wish to modify default settings
   // (please check "src/Config.h" to see all available global options)
   //
   //    (TMVA::gConfig().GetVariablePlotting()).fTimesRMS = 8.0;
   //    (TMVA::gConfig().GetIONames()).fWeightFileDir = "myWeightDirectory";

   // Define the input variables that shall be used for the MVA training
   // note that you may also use variable expressions, such as: "3*var1/var2*abs(var3)"
   // [all types of expressions that can also be parsed by TTree::Draw( "expression" )]
   dataloader->AddVariable("DER_mass_MMC", 'F');
   dataloader->AddVariable("DER_mass_transverse_met_lep", 'F');
   dataloader->AddVariable("DER_mass_vis", 'F');
   dataloader->AddVariable("DER_pt_h", 'F');
   dataloader->AddVariable("DER_deltaeta_jet_jet", 'F');
   dataloader->AddVariable("DER_mass_jet_jet", 'F');
   dataloader->AddVariable("DER_prodeta_jet_jet", 'F');
   dataloader->AddVariable("DER_deltar_tau_lep", 'F');
   dataloader->AddVariable("DER_pt_tot", 'F');
   dataloader->AddVariable("DER_sum_pt", 'F');
   dataloader->AddVariable("DER_pt_ratio_lep_tau", 'F');
   dataloader->AddVariable("DER_met_phi_centrality", 'F');
   dataloader->AddVariable("DER_lep_eta_centrality", 'F');
   dataloader->AddVariable("PRI_tau_pt", 'F');
   dataloader->AddVariable("PRI_tau_eta", 'F');
   dataloader->AddVariable("PRI_tau_phi", 'F');
   dataloader->AddVariable("PRI_lep_pt", 'F');
   dataloader->AddVariable("PRI_lep_eta", 'F');
   dataloader->AddVariable("PRI_lep_phi", 'F');
   dataloader->AddVariable("PRI_met", 'F');
   dataloader->AddVariable("PRI_met_phi", 'F');
   dataloader->AddVariable("PRI_met_sumet", 'F');
   dataloader->AddVariable("PRI_jet_num", 'F');
   dataloader->AddVariable("PRI_jet_leading_pt", 'F');
   dataloader->AddVariable("PRI_jet_leading_eta", 'F');
   dataloader->AddVariable("PRI_jet_leading_phi", 'F');
   dataloader->AddVariable("PRI_jet_subleading_pt", 'F');
   dataloader->AddVariable("PRI_jet_subleading_eta", 'F');
   dataloader->AddVariable("PRI_jet_subleading_phi", 'F');
   dataloader->AddVariable("PRI_jet_all_pt", 'F');

   // You can add so-called "Spectator variables", which are not used in the MVA training,
   // but will appear in the final "TestTree" produced by TMVA. This TestTree will contain the
   // input variables, the response values of all trained MVAs, and the spectator variables
   dataloader->AddSpectator("Weight", "Weight", "", 'F');
   dataloader->AddSpectator("EventId", "EventId", "", 'I');
   dataloader->AddSpectator("Label", "Label", "", 'I');
   dataloader->AddSpectator("KaggleSet", "The Kaggle Set", "", 'c');
   dataloader->AddSpectator("KaggleWeight", "KaggleWeight", "", 'F');


   // global event weights per tree (see below for setting event-wise weights)
   Double_t signalWeight     = 1.0;
   Double_t backgroundWeight = 1.0;

   // You can add an arbitrary number of signal or background trees
   dataloader->AddSignalTree    ( signalTree,     signalWeight );
   dataloader->AddBackgroundTree( bgTree,         backgroundWeight );

   // Set individual event weights (the variables must exist in the original TTree)
   // -  for signal    : `dataloader->SetSignalWeightExpression    ("weight1*weight2");`
   // -  for background: `dataloader->SetBackgroundWeightExpression("weight1*weight2");`
   dataloader->SetBackgroundWeightExpression( "Weight" );

   // Apply additional cuts on the signal and background samples (can be different)
   TCut mycuts = ""; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
   TCut mycutb = ""; // for example: TCut mycutb = "abs(var1)<0.5";

   // Tell the dataloader how to use the training and testing events
   //
   // If no numbers of events are given, half of the events in the tree are used
   // for training, and the other half for testing:
   //
   //    dataloader->PrepareTrainingAndTestTree( mycut, "SplitMode=random:!V" );
   //
   // To also specify the number of testing events, use:
   //
   //    dataloader->PrepareTrainingAndTestTree( mycut,
   //         "NSigTrain=3000:NBkgTrain=3000:NSigTest=3000:NBkgTest=3000:SplitMode=Random:!V" );
   dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,
                                        "nTrain_Signal=1000:nTrain_Background=1000:SplitMode=Random:NormMode=NumEvents:!V" );

   // ### Book MVA methods
   //
   // Please lookup the various method configuration options in the corresponding cxx files, eg:
   // src/MethoCuts.cxx, etc, or here: http://tmva.sourceforge.net/optionRef.html
   // it is possible to preset ranges in the option string in which the cut optimisation should be done:
   // "...:CutRangeMin[2]=-1:CutRangeMax[2]=1"...", where [2] is the third input variable


   // Boosted Decision Trees
   if (Use["BDT"])  // Adaptive Boost
      factory->BookMethod( dataloader, TMVA::Types::kBDT, "BDT",
                           "!H:!V:NTrees=850:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" );


   // For an example of the category classifier usage, see: TMVAClassificationCategory
   //
   // --------------------------------------------------------------------------------------------------
   //  Now you can optimize the setting (configuration) of the MVAs using the set of training events
   // STILL EXPERIMENTAL and only implemented for BDT's !
   //
   //     factory->OptimizeAllMethods("SigEffAt001","Scan");
   //     factory->OptimizeAllMethods("ROCIntegral","FitGA");
   //
   // --------------------------------------------------------------------------------------------------
   //   factory->OptimizeAllMethods();

   // Now you can tell the factory to train, test, and evaluate the MVAs
   //
   // Train MVAs using the set of training events
   factory->TrainAllMethods();

   // Evaluate all MVAs using the set of test events
   factory->TestAllMethods();

   // Evaluate and compare performance of all configured MVAs
   factory->EvaluateAllMethods();

   // --------------------------------------------------------------

   // Save the output
   outputFile->Close();

   std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
   std::cout << "==> Htautau is done!" << std::endl;

   delete factory;
   delete dataloader;
   // Launch the GUI for the root macros
   if (!gROOT->IsBatch()) TMVA::TMVAGui( outfileName );

   return 0;
}


int main( int argc, char** argv )
{
   // Select methods (don't look at this code - not of interest)
   TString methodList;
   for (int i=1; i<argc; i++) {
      TString regMethod(argv[i]);
      if(regMethod=="-b" || regMethod=="--batch") continue;
      if (!methodList.IsNull()) methodList += TString(",");
      methodList += regMethod;
   }
   return Htautau(methodList);
}


Float_t getFloat(TObjArray * dataArray, Int_t idx){
  TObjString * str = (TObjString*)dataArray->At(idx);
  if(str) {
    TString thisData = str->GetString();
    Float_t val = thisData.Atof();
    return val;
  }
  return -999.0;
}

TString getString(TObjArray * dataArray, Int_t idx){
  TObjString * str = (TObjString*)dataArray->At(idx);
  if(str) {
    TString thisData = str->GetString();
    return thisData;
  }
  return "null";
}


TTree * readData(TString fname, TString treename) {
// This function is adapted from the QMUL Practical Machine Learning summer school course to read in the Kaggle 
// data challenge Higgs data from ATLAS.  Specify the CSV file input name, and the function will read the
// data and create a TTree for subsequent use.
//
//The selection applied to data is elementary:
//     1) MMC (H->tautau) mass is calculated for the event
//
//The following variables are available in the kaggle data set
//  0   EventId                           Unique event id for each event - not to be used as a feature
//  1   DER_mass_MMC                      Higgs mass computed using the MMC method
//  2   DER_mass_transverse_met_lep       The m_T between the missing transverse energy and lepton momentum
//  3   DER_mass_vis                      The invariant mass of the Higgs candidate (tau_lep + tau+had system)
//  4   DER_pt_h                          |p_T| of the Higgs candidate.
//  5   DER_deltaeta_jet_jet
//  6   DER_mass_jet_jet
//  7   DER_prodeta_jet_jet
//  8   DER_deltar_tau_lep
//  9   DER_pt_tot
//  10  DER_sum_pt
//  11  DER_pt_ratio_lep_tau
//  12  DER_met_phi_centrality
//  13  DER_lep_eta_centrality
//  14  PRI_tau_pt
//  15  PRI_tau_eta
//  16  PRI_tau_phi
//  17  PRI_lep_pt
//  18  PRI_lep_eta
//  19  PRI_lep_phi
//  20  PRI_met
//  21  PRI_met_phi
//  22  PRI_met_sumet
//  23  PRI_jet_num
//  24  PRI_jet_leading_pt
//  25  PRI_jet_leading_eta
//  26  PRI_jet_leading_phi
//  27  PRI_jet_subleading_pt
//  28  PRI_jet_subleading_eta
//  29  PRI_jet_subleading_phi
//  30  PRI_jet_all_pt
//  31  Weight
//  32  Label
//  33  KaggleSet
//  34  KaggleWeight
  ifstream in(fname);
  if(!in) {
    cout << "unable to open file " << fname << endl;
    return 0;
  }
  TString varList = "";
  in >> varList ;
  cout << "Variable list read in:" << endl;
  cout << varList << endl;
  TObjArray * varArray = varList.Tokenize(',');
  for(int i=0; i < varArray->GetSize()-29; i++) {
    TString thisVar = ((TObjString*)varArray->At(i))->GetString();
    cout << "\t" << i << "\t"<< thisVar << endl;
  }

  cout << "Will now loop over the data to create the tree" << treename << endl;

  TTree *tree = new TTree(treename,"Kaggle Data Tree");

  Float_t EventId(0.);
  Float_t DER_mass_MMC(0.);
  Float_t DER_mass_transverse_met_lep(0.);
  Float_t DER_mass_vis(0.);
  Float_t DER_pt_h(0.);
  Float_t DER_deltaeta_jet_jet(0.);
  Float_t DER_mass_jet_jet(0.);
  Float_t DER_prodeta_jet_jet(0.);
  Float_t DER_deltar_tau_lep(0.);
  Float_t DER_pt_tot(0.);
  Float_t DER_sum_pt(0.);
  Float_t DER_pt_ratio_lep_tau(0.);
  Float_t DER_met_phi_centrality(0.);
  Float_t DER_lep_eta_centrality(0.);
  Float_t PRI_tau_pt(0.);
  Float_t PRI_tau_eta(0.);
  Float_t PRI_tau_phi(0.);
  Float_t PRI_lep_pt(0.);
  Float_t PRI_lep_eta(0.);
  Float_t PRI_lep_phi(0.);
  Float_t PRI_met(0.);
  Float_t PRI_met_phi(0.);
  Float_t PRI_met_sumet(0.);
  Float_t PRI_jet_num(0.);
  Float_t PRI_jet_leading_pt(0.);
  Float_t PRI_jet_leading_eta(0.);
  Float_t PRI_jet_leading_phi(0.);
  Float_t PRI_jet_subleading_pt(0.);
  Float_t PRI_jet_subleading_eta(0.);
  Float_t PRI_jet_subleading_phi(0.);
  Float_t PRI_jet_all_pt(0.);
  Float_t Weight(0.);
  Float_t Label(0.);
  TString sLabel = "";
  Float_t KaggleSet(0.);
  TString sKaggleSet = "";
  Float_t KaggleWeight(0.);

  tree->Branch("EventId",          &EventId, "EventId/F");
  tree->Branch("DER_mass_MMC",     &DER_mass_MMC, "mmc/F");
  tree->Branch("DER_mass_transverse_met_lep", &DER_mass_transverse_met_lep, "mass_transverse_met_lep/F");
  tree->Branch("DER_mass_vis",         &DER_mass_vis, "mass_vis/F");
  tree->Branch("DER_pt_h",             &DER_pt_h, "pt_h/F");
  tree->Branch("DER_deltaeta_jet_jet", &DER_deltaeta_jet_jet, "deltaeta_jet_jet/F");
  tree->Branch("DER_mass_jet_jet",     &DER_mass_jet_jet, "mass_jet_jet/F");
  tree->Branch("DER_prodeta_jet_jet",  &DER_prodeta_jet_jet, "prodeta_jet_jet/F");
  tree->Branch("DER_deltar_tau_lep",   &DER_deltar_tau_lep, "deltar_tau_lep/F");
  tree->Branch("DER_pt_tot",           &DER_pt_tot, "pt_tot/F");
  tree->Branch("DER_sum_pt",           &DER_sum_pt, "sum_pt/F");
  tree->Branch("DER_pt_ratio_lep_tau",   &DER_pt_ratio_lep_tau, "DER_pt_ratio_lep_tau/F");
  tree->Branch("DER_met_phi_centrality", &DER_met_phi_centrality, "DER_met_phi_centrality/F");
  tree->Branch("DER_lep_eta_centrality", &DER_lep_eta_centrality, "DER_lep_eta_centrality/F");
  tree->Branch("PRI_tau_pt",       &PRI_tau_pt, "PRI_tau_pt/F");
  tree->Branch("PRI_tau_eta",      &PRI_tau_eta, "PRI_tau_eta/F");
  tree->Branch("PRI_tau_phi",      &PRI_tau_phi, "PRI_tau_phi/F");
  tree->Branch("PRI_lep_pt",       &PRI_lep_pt, "PRI_lep_pt/F");
  tree->Branch("PRI_lep_eta",      &PRI_lep_eta, "PRI_lep_eta/F");
  tree->Branch("PRI_lep_phi",      &PRI_lep_phi, "PRI_lep_phi/F");
  tree->Branch("PRI_met",          &PRI_met, "PRI_met/F");
  tree->Branch("PRI_met_phi",      &PRI_met_phi, "PRI_met_phi/F");
  tree->Branch("PRI_met_sumet",    &PRI_met_sumet, "PRI_met_sumet/F");
  tree->Branch("PRI_jet_num",      &PRI_jet_num, "PRI_jet_num/F");
  tree->Branch("PRI_jet_leading_pt",     &PRI_jet_leading_pt, "PRI_jet_leading_pt/F");
  tree->Branch("PRI_jet_leading_eta",    &PRI_jet_leading_eta, "PRI_jet_leading_eta/F");
  tree->Branch("PRI_jet_leading_phi",    &PRI_jet_leading_phi, "PRI_jet_leading_phi/F");
  tree->Branch("PRI_jet_subleading_pt",  &PRI_jet_subleading_pt, "PRI_jet_subleading_pt/F");
  tree->Branch("PRI_jet_subleading_eta", &PRI_jet_subleading_eta, "PRI_jet_subleading_eta/F");
  tree->Branch("PRI_jet_subleading_phi", &PRI_jet_subleading_phi, "PRI_jet_subleading_phi/F");
  tree->Branch("PRI_jet_all_pt",   &PRI_jet_all_pt, "PRI_jet_all_pt/F");
  tree->Branch("Weight",           &Weight, "Weight/F");
  tree->Branch("Label",            &Label, "Label/F");
  tree->Branch("KaggleSet",        &KaggleSet, "KaggleSet/F");
  tree->Branch("KaggleWeight",     &KaggleWeight, "KaggleWeight/F");
 
  while(!in.eof()) {
    // loop over the data to read in events for processing into signal and background trees
    //    cout << "Reading event " << j << endl;
    
    TString thisData = "";
    in >> thisData;
    TObjArray * dataArray = thisData.Tokenize(',');
    if(dataArray && (dataArray->GetSize() == 64)){
      DER_mass_MMC           = getFloat(dataArray, 1);
      DER_mass_transverse_met_lep = getFloat(dataArray, 2);
      DER_mass_vis           = getFloat(dataArray, 3);
      DER_pt_h               = getFloat(dataArray, 4);
      DER_deltaeta_jet_jet   = getFloat(dataArray, 5);
      DER_mass_jet_jet       = getFloat(dataArray, 6);
      DER_prodeta_jet_jet    = getFloat(dataArray, 7);
      DER_deltar_tau_lep     = getFloat(dataArray, 8);
      DER_pt_tot             = getFloat(dataArray, 9);
      DER_sum_pt             = getFloat(dataArray, 10);
      
      EventId                = getFloat(dataArray, 0);
      
      DER_pt_ratio_lep_tau   = getFloat(dataArray, 11);
      DER_met_phi_centrality = getFloat(dataArray, 12);
      DER_lep_eta_centrality = getFloat(dataArray, 13);
      PRI_tau_pt             = getFloat(dataArray, 14);
      PRI_tau_eta            = getFloat(dataArray, 15);
      PRI_tau_phi            = getFloat(dataArray, 16);
      PRI_lep_pt             = getFloat(dataArray, 17);
      PRI_lep_eta            = getFloat(dataArray, 18);
      PRI_lep_phi            = getFloat(dataArray, 19);
      PRI_met                = getFloat(dataArray, 20);
      PRI_met_phi            = getFloat(dataArray, 21);
      PRI_met_sumet          = getFloat(dataArray, 22);
      PRI_jet_num            = getFloat(dataArray, 23);
      PRI_jet_leading_pt     = getFloat(dataArray, 24);
      PRI_jet_leading_eta    = getFloat(dataArray, 25);
      PRI_jet_leading_phi    = getFloat(dataArray, 26);
      PRI_jet_subleading_pt  = getFloat(dataArray, 27);
      PRI_jet_subleading_eta = getFloat(dataArray, 28);
      PRI_jet_subleading_phi = getFloat(dataArray, 29);
      PRI_jet_all_pt         = getFloat(dataArray, 30);
      Weight                 = getFloat(dataArray, 31);
      sLabel                 = getString(dataArray, 32);
      sKaggleSet             = getString(dataArray, 33);
      KaggleWeight           = getFloat(dataArray, 34);
    }
    if(sLabel.Contains("s")) Label = 1;
    else Label = 0;

    // b, t, u, v are the KaggleSet options - stor these as 0, 1, 2, 3
    if(sKaggleSet.Contains("b")) KaggleSet = 0;       // public leaderboard
    else if(sKaggleSet.Contains("t")) KaggleSet = 1;  // train
    else if(sKaggleSet.Contains("u")) KaggleSet = 2;  // unused
    else if(sKaggleSet.Contains("v")) KaggleSet = 3;  // private leaderboard

    Bool_t isGood = kTRUE;
    if(DER_mass_MMC < 0)  isGood = kFALSE;

    if(isGood) {
	tree->Fill();
    }
  }

  tree->Print("v");

  cout << "After processing the data there are" << endl;
  cout << "\t" << tree->GetEntries() << " signal" << endl;
  return tree;
}
