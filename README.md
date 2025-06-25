# AMLGentex

AMLGentex is a benchmarking suite developed by AI Sweden in collaboration with Handelsbanken and Swedbank. It supports the generation of realistic synthetic transaction data, the training of machine learning models, and the application of explainability techniques. The project aims to provide researchers and practitioners with accessible, high-quality data to advance the development and evaluation of anti-money laundering systems. In particular, the data is loosely based on the mobile payment system SWISH but is easily extended beyond. As illustrated in the figure below, AMLGentex captures a range of real-world data complexities, each assessed in terms of severity by AML experts from Swedbank and Handelsbanken.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ff9c8a04-2344-4f5d-8821-f623464c826c" 
       width="400" 
       alt="Key challenges in transaction monitoring for AML based on expert opinions from AML practitioners">
</p>

## Accronyms and definitions
* SAR: Suspicious Activity Report - accounts or transactions that are labeled as suspisious by the bank
* SWISH: Swedish Instant Payment System - a mobile payment system used in Sweden
* AML: Anti-Money Laundering - the process of detecting and preventing money laundering
* Transaction: A SWISH transaction between two accounts
* Income: A transaction from a source to and an account (not a SWISH transaction)
* Outcome: A transaction from an account to a sink (not a SWISH transaction)

## Synthetic Data Generation
The synthetic data generation is inspired by IBM's AMLSim (https://github.com/IBM/AMLSim) but with extensions in multiple directions. In particular, AMLGentex offers
* A controlled way to generate scale-free transaction networks (https://en.wikipedia.org/wiki/Scale-free_network) based on three parameters. This provides the user with great control over the in- and out-degree distribution in the network
* A matching mechanism (heavily updated from AMLSim) where user-defined patterns (both normal and laundering) are sequentially inserted into the network. The normal patterns are only inserted if there is room in the blueprint network (based on the degree distribution) whereas the money laundering patterns are always included. The process is outlined in the figure below.
* Modelling of in- and out-flow of funds in the network. Funds enter the network by means of salaries and goes out by accounts performing transactions. The income and outcome nodes are denoted as the source and the sink, respectively.
* A simple statistical model for account behaviour that keeps track of the average balance over a sliding window which determines if the account is likely to spend in the current time. A balance above the average indicates a sense of being resourceful and hence more likely to spend.
* Money laundering occurs in a three-stage process: placement, layering, integration.
* Accounts engaging in money laundering have a user-defined probability of engaging in multiple laundering events.

<p align="center">
  <img width="681" alt="Screenshot 2025-06-25 at 17 13 47" src="https://github.com/user-attachments/assets/85994420-754d-47ff-8107-39bab914d835" />
</p>



# AMLsim
AMLsim is a simulator for generating transaction networks used in anti-money laundering research. It is based on the simulator by IBM (TODO: add link) and is extended to utilize distributions and model behavioural features. This version is designed to generate SWISH data of personal accounts. It can simulate income and outcome for accounts, as well as known transactions patterns of normal and suspisious behaviour. In short, it has two parts: a python part for generating the transaction network and a java part for simulating the behaviour of the agents. The simulation is controlled by 6 parameter files:
* 1 json file, which defines behviours of accounts and some paths varibles used during the simulation. 
* 5 csv files, which defines some inital condtions and together defines the structure of the transaction network.

The output of the simulation is a csv file with all the transactions.


## Dependencies

Dependencies: python3.7, java, maven

1. clone repo
2. move into AMlsim folder
3. install python dependencies: `pip install -r requirements.txt` or `conda env create -f AMLamlsim.yml`
4. install java dependencies: `mvn install:install-file -Dfile=jars/mason.20.jar -DgroupId=mason -DartifactId=mason -Dversion=20 -Dpackaging=jar -DgeneratePom=true`

## Setup

1. Create a folder for the outputs: `mkdir outputs`
2. Create a temporary folder for storing pyhton output: `mkdir tmp`
3. Create a folder for the simulation paramters: `mkdir paramFiles`
4. In paramFiles create a folder for a new simulation, e.g. `mkdir paramFiles/simulation1`
5. In the simulation folder, create these files: conf.json, accounts.csv, normalModels.csv, alertPatterns.csv, degree.csv and transactionTypes.csv
6. Specify the parameters in the files (see below)

## Specify parameters (with examples)

### conf.json

The conf.json file contains parameters for the generel behaviour of the accounts and paths to the other files, the paths are relative to the conf.json file. A example looks like this:
```
{
  "general": {
    "random_seed": 0,
    "simulation_name": "simulation1",
    "total_steps": 86
  },
  "default": {
    "min_amount": 1,
    "max_amount": 150000,
    "mean_amount": 637,
    "std_amount": 1000,
    "mean_amount_sar": 2000,
    "std_amount_sar": 1000,
    "prob_income": 0.05,
    "mean_income": 500.0,
    "std_income": 1000.0,
    "prob_income_sar": 0.05,
    "mean_income_sar": 500.0,
    "std_income_sar": 1000.0,
    "mean_outcome": 200.0,
    "std_outcome": 500.0,
    "mean_outcome_sar": 200.0,
    "std_outcome_sar": 500.0,
    "prob_spend_cash": 0.7,
    "n_steps_balance_history": 56,
    "mean_phone_change_frequency": 1460,
    "std_phone_change_frequency": 365,
    "mean_phone_change_frequency_sar": 365,
    "std_phone_change_frequency_sar": 182,
    "mean_bank_change_frequency": 1460,
    "std_bank_change_frequency": 365,
    "mean_bank_change_frequency_sar": 365,
    "std_bank_change_frequency_sar": 182,
    "margin_ratio": 0.1,
    "prob_participate_in_multiple_sars": 0.1
  },
  "input": {
    "directory": "paramFiles/simulation1",
    "schema": "schema.json",
    "accounts": "accounts.csv",
    "alert_patterns": "alertPatterns.csv",
    "normal_models": "normalModels.csv",
    "degree": "degree.csv",
    "transaction_type": "transactionType.csv",
    "is_aggregated_accounts": true
  },
  "temporal": {
    "directory": "tmp",
    "transactions": "transactions.csv",
    "accounts": "accounts.csv",
    "alert_members": "alert_members.csv",
    "normal_models": "normal_models.csv"
  },
  "output": {
    "directory": "outputs",
    "transaction_log": "tx_log.csv"
  },
  "graph_generator": {
    "degree_threshold": 1
  },
  "simulator": {
    "transaction_limit": 100000,
    "transaction_interval": 7,
    "sar_interval": 7
  },
  "scale-free": {
    "gamma": 2.0,
    "loc": 1.0,
    "scale": 1.0
  }
}
```
* **random_seed**

    The random seed is used to make the simulation reproducable.

* **simulation_name**

    The name of the simulation, used for naming the tmp and output folder.

* **total_steps**

    The total number of steps in the simulation. A step is not tied to a specific time unit. However, we typically view one step as a day.

* **min_amount, max_amount, mean_amount, std_amount, mean_amount_sar, std_amount_sar**

    The min and max amount of a transaction, and the mean and standard deviation of the truncated normal distribution used to sample the amount of a transaction, see Fig. 1. The distribution is truncated to zero and current blanace of the account. Mean and std are specifed for normal and SAR transactions, respectively.

    <div align="center">
    <img src="https://github.com/aidotse/flib/blob/main/resources/gaussian.jpeg" alt="Fig 1. Truncated Gaussian distribution" width="500" height="300">
    <br>
    <em>Fig 1. Truncated Gaussian distribution.</em>
    </div>

* **prob_income, mean_income, std_income, prob_income_sar, mean_income_sar, std_income_sar**

    * These variables are used to set the in-flux of money to the network beyond salaries.  
  
    * The probability for an account to recive a transaction in a given time step is given by **prob_income**.
    Note that this transaction is coming from the source.

    * The size of the transaction is sampled from a truncated Gaussian distribution with mean **mean_income** and standard deviation **std_income**, see Fig. 1.
 
    * Influx of money to SAR accounts are handled similarly via **prob_income_sar**, **mean_income_sar**, and **std_income_sar**
 
    

* **mean_outcome, std_outcome, mean_outcome_sar, std_outcome_sar, n_steps_balance_history**
  * The **mean_outcome** and **std_outcome** denote the parameters for a truncated Gaussian distribution used to sample the size of the spending transactions, i.e., transactions going to the sink.
  * **mean_outcome_sar** and **std_outcome_sar** are similarly parametrizing the sampling distribution for spendings of SAR accounts.
  * Each account behave individually depending on its balance history. In particular, for each account, the probability of spending in step $t$ is obtained by first deciding if the account is "feeling rich or poor" as
    $$d_t = \left( x_i - \frac{1}{N}\sum_{j=i-N}^{i} x_j  \right) \text{\huge/} \frac{1}{N}\sum_{j=i-N}^{t} x_j$$
    where $x_t$ is the account balance in the current time step $t$ and $N$ (n_steps_balance_history) is the number of past steps considered.
    The spending probability is then obtained from a Sigmoid function as $p_t = 1 / (1 + e^{-d_t})$ whereafter a transaction to the sink is performed if $y=1$ where $y\sim\mathrm{Ber}(p_t)$. See Fig. 2 for an illustration of the procedure.

     <div align="center">
    <img src="https://github.com/aidotse/flib/blob/main/resources/spending.jpeg" alt="Fig 1. Truncated Gaussian distribution" width="400" height="300">
    <br>
    <em>Fig 2. Spending behavior.</em>
    </div>

* **prob_spend_cash**

    The probability for an account to spend cash. If an account has cash, **prop_spend_cash** will decide if the account spends cash in a outcome transaction. Only sar accounts can spend cash.

* **mean_phone_change_frequency, std_phone_change_frequency, mean_phone_change_frequency_sar, std_phone_change_frequency_sar**

  * **mean_phone_change_frequency** and **std_phone_change_frequency** are used to sample from a truncated Gaussian distribution (rounded to nearest integer) to decide the number of days before an account will change the phone number.
  * **mean_phone_change_frequency_sar** and **std_phone_change_frequency_sar** serve a similar function as above but for SAR accounts.

* **mean_bank_change_frequency, std_bank_change_frequency, mean_bank_change_frequency_sar, std_bank_change_frequency_sar**

  * **mean_bank_change_frequency** and **std_bank_change_frequency**  are used to sample from a truncated Gaussian distribution (rounded to nearest integer) to decide the number of days before an account will change bank.
 
  * **mean_bank_change_frequency_sar** and **std_bank_change_frequency_sar** serve a similar function as above but for SAR accounts.

* **margin_ratio**

    * Whenever an account engages in money laundering, it takes a cut (percentage) of the money given by **margin_ratio**. 

* **prob_participate_in_multiple_sars**

  * The probability for an sar account to participate in multiple SAR patterns. The number of SAR patterns an account can participate in follows the probability mass function of a Log($p$)-distributed random varible
    $$\mathrm{Pr}[k] = \frac{-p^k}{k\log(1-p)}$$,
    where $k$ is the number of SAR patterns an account participates in and $p$ is the parameter **prob_participate_in_multiple_sars**.

* **gamma**, **loc**, and **scale**
  * These parameters are used to generate a [scale-free network](https://en.wikipedia.org/wiki/Scale-free_network)
  * For large degrees $d$, a scale-free network obeys $\mathrm{Pr}[\text{node degree}=d] \propto d^{-\gamma}$ where $\gamma$ is given by **gamma**
  * **loc** and **scale** are used to control the degree distribution for smaller values of $d$, i.e., before the power law kicks in
  * See Fig. 3 for a visualization.
  <div align="center">
  <img src="https://github.com/aidotse/flib/blob/main/resources/degree.jpeg" alt="Fig 1. Truncated Gaussian distribution" width="400" height="300">
  <br>
  <em>Fig 3. Parameters for the scale-free network.</em>
  </div>
    

### account.csv

The accounts.csv file contains the initial conditions for the accounts. It has the following columns:
* **count**: (int) The number of accounts to generate.
* **min_balance, max_balance**: (int) The minimum and maximum inital balance of the accounts. The inital balance is sampled from a uniform distribution.
* **country**: (string) The country of the accounts. 
* **business_type**: (string) The type of business of the accounts, OBS: currently only "I" is supported.
* **bank**: (string) The bank of the accounts.

Below is an example where 1000 accounts are generated in two banks with groups of users starting with different initial balances.
``` 
count,min_balance,max_balance,country,business_type,bank
200,1000,10000,SWE,I,bank_a
200,10000,100000,SWE,I,bank_a
100,100000,200000,SWE,I,bank_a
200,1000,10000,SWE,I,bank_b
200,10000,100000,SWE,I,bank_b
100,100000,200000,SWE,I,bank_b
```

### normalModels.csv

normalModels.csv contains the normal transaction-patterns of the accounts. It has the following columns:
* **count**: (int) The number of patterns to generate.
* **type**: (string) The type of the pattern. Can be single, fan_out, fan_in, forward, mutual or periodical. Se below for pattern definitions.
* **schedule_id**: (int) The id of the schedule to use for the pattern. Can be 0, 1, 2 or 3. Se below for schedule definitions.
* **min_accounts, max_accounts**: (int) The minimum and maximum number of accounts in the pattern. The simulator will find subsets of accounts where the pattern fits and sample from these. The number of subsets will depend on the min and max and on the structure of the network, defined in degree.csv. Some patterns has a fixed number of accounts, se pattern definition for more information.
* **min_period, max_period**: (int) The minimum and maximum period of the pattern. The period is the number of steps for a pattern to complet. The period is sampled from a uniform distribution.
* **bank_id**: (int) If specified, the patterns will only include accounts from that bank. If left blank, the patterns can include accounts from any bank specified within accounts.csv.

Below is an example where 2000 different patterns are generated with varying number of accounts and periods. Some patterns are restricted to a specific bank, while others can include several banks.
```
count,type,schedule_id,min_accounts,max_accounts,min_period,max_period,bank_id
100,single,0,2,2,1,84,bank_a
100,fan_in,1,6,8,21,21,bank_a
100,fan_out,2,6,10,7,14,bank_a
100,periodical,3,2,2,1,84,bank_a
100,single,0,2,2,1,84,bank_b
100,fan_in,1,6,8,21,21,bank_b
100,fan_out,2,6,10,7,14,bank_b
100,periodical,3,2,2,1,84,bank_b
300,forward,2,3,3,2,4,
300,mutual,2,2,2,1,10,
300,fan_out,2,12,16,28,56,
300,fan_in,2,10,20,56,84
```

### alertPatterns.csv
alertPatterns.csv contains the suspisious transaction patterns of the accounts. In contrast to normal models, the alert patterns will be place on top of the normal model transaction network. First the normal models will build a graph according to the degree.csv. Alert patterns will then be add into the graph, completly ignoring the degree.csv file. See (TODO: add section that clarifies this) for more information. alertPatterns.csv has the following columns:
* **count**: (int) The number of patterns to generate.
* **type**: (string) The type of the pattern. Can be fan_out, fan_in, cycle, random, bipartite, stack, gather_scatter or scatter_gather. Se below for pattern definitions.
* **schedule_id**: (int) The id of the schedule to use for the pattern. Can be 0, 1, 2 or 3. Se below for schedule definitions.
* **min_accounts, max_accounts**: (int) The minimum and maximum number of accounts in the pattern sampled from a uniform distribution. Some patterns has a minumum number of accounts, se pattern definition for more information. 
* **min_amount, max_amount**: (int) OBS: not used! 
* **min_period, max_period**: (int) The minimum and maximum period of the pattern. The period is the number of steps for a pattern to complet. The period is sampled from a uniform distribution.
* **bank_id**: (int) If specified, the patterns will only include accounts from that bank. If left blank, the patterns can include accounts from all banks.
* **is_sar**: (string) Can be true or false. If true, the pattern will be labeled as suspisious. If false, the pattern will be labeled as normal.
* **source_type**: (string) Can be CASH or TRANSFER. See (TODO: add section that clarifies this) for more information.

Below is an example with 8 alert patterns:
```
count,type,schedule_id,min_accounts,max_accounts,min_amount,max_amount,min_period,max_period,bank_id,is_sar,source_type
1,fan_out,2,10,20,100,1000,14,28,bank_a,True,CASH
1,fan_in,2,110,20,100,1000,14,28,bank_b,True,TRANSFER
1,cycle,2,10,20,100,1000,14,28,,True,TRANSFER
1,random,2,10,20,100,1000,14,28,,True,CASH
1,bipartite,2,10,20,100,1000,14,28,,True,TRANSFER
1,stack,2,10,20,100,1000,14,28,,True,TRANSFER
1,gather_scatter,2,10,20,100,1000,14,28,,True,CASH
1,scatter_gather,2,10,20,100,1000,14,28,,True,CASH
```

### degree.csv
The degree.csv file defines the structure of the transaction network and can be automatically generated by the script generate_scalefree.py. It has the following columns:
* **count**: (int) The number of nodes, aka accounts.
* **In-degree**: (int) The in-degree of the nodes.
* **Out-degree**: (int) The out-degree of the nodes.

The graph needs to be complete, i.e. the sum of the in-degree and out-degree of all nodes needs to be equal. Further, the total count needs to be equal to the number of accounts in the accounts.csv file. Below is an example for a graph with 1000 nodes.
```
count,In-degree,Out-degree
512,10,10
256,20,20
128,30,30
64,40,40
32,50,50
16,60,60
8,70,70
4,80,80
2,90,90
```

### transactionType.csv
transactionType.csv defines avalible transaction types. OBS: CURRENTLY ONLY ONE TYPE IS IMPLEMENTED! It has the following columns:
* **Type**: (string) The type of the transaction. Can only be TRANSFER. 
* **Frequency**: (int) The frequency of the transaction type used in the transaction network.

Write this in the transactionType.csv file:
```
Type,Frequency
TRANSFER,1
```

### Network creation
The network is created according to the following procedure:
  * **degree.csv** is used as a blueprint for the transaction network
  * The patterns defined in **normalModels.csv** are iterated and injected into the bluepring network.
  * Once all the normal transactions are injected into the network, the alert patterns in **alertPatterns.csv** are injected by randomly assigning nodes to become SAR accounts.
  Note that nodes that are not included in a pattern will be discarded. See Fig. 4 for an illustration of the procedure.
    <div align="center">
    <img src="https://github.com/aidotse/flib/blob/main/resources/pattern.jpeg" alt="Fig 4. Truncated Gaussian distribution" width="600" height="300">
    <br>
    <em>Fig 4. Network creation.</em>
    </div>

## Run


### Alternative 1: Docker
Run the docker image with the following command:
```
docker run -v /path/to/paramFiles:/app/paramFiles -v /path/to/outputs:/app/outputs thecoldice/amlsim:latest /path/to/conf.json
```

### Alternative 2: Manual
1. Run the python script: `python scripts/transaction_graph_generator.py /path/to/conf.json`
2. Run the java program: `java -jar target/AMLSim-1.0-SNAPSHOT.jar /path/to/conf.json` (unclear if this is will work...)

## Pattern definitions
To be added.

## Schedule definitions
The schedual id is used to specify how transactions within a pattern will occur in the temporal dimension. A pattern with more the one transaction can happen over several steps according to a predefined pattern. The schedule id is used to specify this pattern. Below are the four different schedules:
* **Fixed interval**: id = 0. Each transactions in the pattern will occur sequentially every $k$ time step where $k$ is given by the interval specified in the conf.json file. 
* **Random interval**: id = 1. A random interval will be generated uniformly within the provided period. Each transactions in the pattern will then occur sequentially.
* **Unorderd**: id = 2. The transactions in the pattern will be placed in a random order over the period of the pattern.
* **Simultaneous**: id = 3. All transactions in the pattern will occur at the same time step.


---

# 
If you use AMLGentex in your work, please cite the following paper:

```bibtex
@misc{ostman2025amlgentexmobilizingdatadrivenresearch,
  title     = {AMLgentex: Mobilizing Data-Driven Research to Combat Money Laundering},
  author    = {Johan \"Ostman and Edvin Callisen and Anton Chen and Kristiina Ausmees and Emanuel G\aardh and Jovan Zamac and Jolanta Goldsteine and Hugo Wefer and Simon Whelan and Markus Reimeg\aard},
  year      = {2025},
  eprint    = {2506.13989},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SI},
  url       = {https://arxiv.org/abs/2506.13989}
}

