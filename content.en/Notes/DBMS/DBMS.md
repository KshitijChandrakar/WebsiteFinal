---
date: 2024-08-09T10:58:08-04:00
categories: 'Notes'
tags: [semester3, DBMS]
title: "DBMS August 9"
weight: 2
---
#databases #semester3  
### Reference
[Oracle Docs](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/index.html)
[W3Schools Reference](https://www.w3schools.com/sql/)

#### Characteristics of Database Approach
- Main Characteristics of Database Approach
	1. Self Describing nature of a DB System [ Catalog of the Data ]
		- it contains the DB and Metadata
		- The DB Catalog is Used by
			-  DBMS Software
			- DB Users
		###### Comparision with Traditional File Processing (TFP)
			TFP can only access specific DBs
		1. Insulation between program  and data abstraction
			-  Program Data Independence - structure of the data files is stored in dbms catalog separatey from access programs
			- VS TFP Any Change to the Structure should be reflect in the corresponding programs
		2. Program operation Independence
			- In DDDBMS and ORDBMS users can define operations on data as part of DB Definitions
			- an operation is specified in two parts
			- implementation of the operation is specified separately and can be changed without affecting the instance
		2. Data Abstraction
			- Allows program data independence and Program Operation Independence
			- It provides users with conceptual representation of data
			-  Supports Multiple Views of Data
		3. Supports Multiuser Transaction Processing
			- Multiuser DBMS
			- Concorrency Control Software
			- Online Transaction Processing Application
