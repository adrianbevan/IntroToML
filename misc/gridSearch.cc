Double_t true_m = 1.0;
Double_t true_c = 1.0;
const Int_t N = 100;
Double_t x[N];
Double_t y[N];

Double_t lossFunction(Double_t this_m, Double_t this_c, Double_t *x){
  Double_t lossSum = 0.0;
  for(Int_t i=0; i < N; i++){
    Double_t yhat= this_m*x[i] + this_c;
    lossSum += (yhat - y[i])*(yhat - y[i]);
  }
  return lossSum;
}

void gridSearch() {
  // generate the data to fit
  Step = 2.0/N;
  for(Int_t i=0; i < N; i++){
    x[i] = Step*i;
    y[i] = true_m*x[i] + true_c;
  }

  TH2F hist("h", "", 20, 0.0, 2.0, 20, 0.0, 2.0);
  hist.GetXaxis()->SetTitle("#hat{m}");
  hist.GetYaxis()->SetTitle("#hat{c}");
  hist.GetZaxis()->SetTitle("L_{2} loss");
  hist.GetXaxis()->SetTitleOffset(1.6);
  hist.GetYaxis()->SetTitleOffset(1.6);
  hist.GetZaxis()->SetTitleOffset(1.2);

  hist.GetXaxis()->CenterTitle();
  hist.GetYaxis()->CenterTitle();
  hist.GetZaxis()->CenterTitle();

  Double_t minL2 = 1e6;
  Double_t min_c = -999.0;
  Double_t min_m = -999.0;

  // loop over j (steps in m) and k (steps in c)
  for(Int_t k = 0; k < 20; k++){
    Double_t this_c = 0.1*k;
    for(Int_t j = 0; j < 20; j++){
      Double_t this_m = 0.1*j;
      Double_t thisLoss = lossFunction(this_m, this_c, x);
      hist.Fill(this_m, this_c, thisLoss);

      if(thisLoss < minL2) {
	minL2 = thisLoss;
	min_c = this_c;
	min_m = this_m;
      }
      cout << minL2 << " m = " << this_m << " c = " << this_c << endl;
    }
  }

  cout << "Optimal values" << endl;
  cout << " m  = " << min_m << endl;
  cout << " c  = " << min_c << endl;
  cout << " L2 = " << minL2 << endl;

  hist.SetStats(0);
  TCanvas can("can");
  can.Draw();
  hist.Draw("surf");
  can.Print("gridSearch.pdf");

  TFile f("gridSearch.root", "RECREATE");
  f.cd();
  hist.Write();
  f.Write();
  f.Close();
}
