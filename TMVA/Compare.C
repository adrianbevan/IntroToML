#include "TFile.h"
#include "TCanvas.h"
#include "TH1F.h"
#include <iostream>

void Compare()
{
  std::cout << "Running comparison of ROC curves for models" << std::endl;
  TFile *f1 = new TFile("TMVA-unoptimised.root");
  TFile *f2 = new TFile("TMVA-optimised.root");
  TFile *f3 = new TFile("TMVA-deco.root");
  TFile *f4 = new TFile("TMVA-decoOpt.root");

  Bool_t good = kTRUE;
  if(!f1)  { std::cout << "\tUnable to open file TMVA-unoptimised.root" << std::endl; good = kFALSE; }
  if(!f2)  { std::cout << "\tUnable to open file TMVA-optimised.root" << std::endl;   good = kFALSE; }
  if(!f3)  { std::cout << "\tUnable to open file TMVA-deco.root" << std::endl;        good = kFALSE; }
  if(!f4)  { std::cout << "\tUnable to open file TMVA-decoOpt.root" << std::endl;     good = kFALSE; }

  if(!good) return;

  TH1F* h1 = (TH1F*)f1->Get("dataset/Method_BDT/BDT/MVA_BDT_trainingRejBvsS");
  TH1F* h2 = (TH1F*)f2->Get("dataset/Method_BDT/BDT/MVA_BDT_trainingRejBvsS");
  TH1F* h3 = (TH1F*)f3->Get("dataset/Method_BDT/BDT/MVA_BDT_trainingRejBvsS");
  TH1F* h4 = (TH1F*)f4->Get("dataset/Method_BDT/BDT/MVA_BDT_trainingRejBvsS");

  if(!h1)  { std::cout << "\tUnable to get ROC curve from TMVA-unoptimised.root" << std::endl;  good = kFALSE; }
  if(!h2)  { std::cout << "\tUnable to get ROC curve from TMVA-optimised.root" << std::endl;    good = kFALSE; }
  if(!h3)  { std::cout << "\tUnable to get ROC curve from TMVA-deco.root" << std::endl;         good = kFALSE; }
  if(!h4)  { std::cout << "\tUnable to get ROC curve from TMVA-decoOpt.root" << std::endl;      good = kFALSE; }

  if(!good) return;

  if(good) {
    TCanvas can("can", "");
    can.cd();
    if(h1) {
      h1->SetStats(0);
      h1->GetXaxis()->SetTitle("#epsilon_{signal}");
      h1->GetYaxis()->SetTitle("1 - #epsilon_{background}");
      h1->Draw();
    }
    if(h2) {
      h2->SetLineColor(kRed); 
      h2->Draw("same");
    }
    if(h3) {
      h3->SetLineColor(kBlue); 
      h3->Draw("same");
    }
    if(h4) {
      h4->SetLineColor(kGreen); 
      h4->Draw("same");
    }
    TLegend leg(0.2, 0.2, 0.5, 0.5);
    leg.AddEntry(h1, "Out Of The Box");
    leg.AddEntry(h2, "Optimised");
    leg.AddEntry(h3, "OOTB-Decorrelated");
    leg.AddEntry(h4, "Decorrelated and Opt.");
    leg.Draw();

    can.Print("Compare.pdf");
  }
}
