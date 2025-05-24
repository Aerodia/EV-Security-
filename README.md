# EV-Security-
Machine learning implementation on CICEVSE Dataset
This repository contains our research, code, and analysis for our project: "Malware Detection in EV Charging Infrastructure using Network Traffic and HPC Kernel Events." Our project proposes a multi-layer approach of securing EV charging stations using an ML-based methodology of combining external network traffic information with internal system-level event logs.

1.Project Overview
With increased adoption of EVs, securing the Electric Vehicle Supply Equipment (EVSE) against advanced cyberattacks is also a growing need. Most conventional approaches are anomaly detection in the network-based sense only. Our project takes it a step further and combines kernel-level Hardware Performance Counter (HPC) events with network traffic logs to identify malicious activity in a more effective way.
We utilized the CICEVSE2024 dataset, an open rich dataset that includes:
1.Full bidirectional network traffic logs (.pcap and CSV)
2.Kernel-level and HPC traces in attack and normal modes
3.Data on energy consumption during various cyberattacks

2.Technologies Used
  Python 3.17
  Scikit-learn
  Pandas, NumPy
  NFStream (to convert .pcap to CSV flow-based)
  Linux perf tools (kernel & HPC events – data collected outside)

3.Models Assessed
We tested and compared certain machine learning models for binary and multi-class classification, including:
  XGBoost – Maximum accuracy (99.2%) in network traffic identification
  Random Forest, MLP – High performance (98.6–98.9%)
  SVM – Robust and effective in high-dimensional binary problems
  KNN – Highest recall in kernel event classification
  AdaBoost – Good recall but higher number of false positives
  Logistic Regression, Naive Bayes – Less accurate and lightweight for complex tasks
