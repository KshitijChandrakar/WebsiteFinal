---
date: 2024-08-14T10:58:08-04:00
categories: 'Notes'
tags: [semester3, DBMS]
title: "DBMS August 14"
---
```SQL
create table customers (
    cid int constraint pk primary key,
    customer_name varchar(20) check (customer_name is NOT null)
)

insert into customers (cid, customer_name) values(1, 'name')
```

## Adding a name to the constraint
```SQL
create table customers (
    cid int constraint pk primary key,
    customer_name varchar(20) constraint ck check (customer_name is NOT null)
)

insert into customers (cid, customer_name) values(1, 'name')
```

## Dropping the constraint
```SQL
Alter TABLE customers drop constraint ck
```
![[Pasted image 20240814143949.png]]

## Seeing the constrains in the database
```SQL
select constraint_name
from user_constraints
where table_name = 'CUSTOMERS'
```
![[Pasted image 20240814144729.png]]

## On Foreign Keys of a secondary table
- The Domain Should be the Same as the primary key of the primary table
- The FK should *reference* a value from the primary table (ie there should be a value in the primary table which is the same you tried to enter into the secondary table)
- it can refer to another tuple in its own relation
```SQL
--Rules while working with FKs
--Example 1

create table t1(
    deptid number primary key,
    debtname varchar(20)
)
insert into t1 values (100,'police')
insert into t1 values (200, 'Hospital')

--Rule 1: Attributes of fk should have same domain as of referred pk

create table t2(
    empid number primary key,
    deptid number,
    name varchar(20),
    foreign key (deptid) references t1(deptid) --It References deptid from t1
)
desc t2
--Rule 2; vlaue of fk =value of referred pk (or) null
insert table t2 values (1,100,'Alex')
insert table t2 values (2,100,'Beno')

--Rule 3: fk can refer to another tuple in its own relation

```

 - Inserting a value into t2 fk would also add it to the fk table
	 - thus you need to insert in a correct order

```SQL
create table t3(
    empid number primary key,
    empname varchar(20),
    managerid number,
    foreign key (managerid) references t1(debtit)
)
    --ERROR (wrong order inserting records)

    insert all
    into t3 values (1,'Ayush', 3)
    into t3 values (1,'Ayus', 3)
    into t3 values (1,'Ayu', 5)
    into t3 values (1,'Ay', 5
    into t3 values (1,'A', 6)
    into t3 values (1,'Ayushi', 6)
select * from dual

--NO ERROR(correct order) (DESC ORDER cause the 6 record needs to exist as 5 would reference 6)
Insert all
into t3 values (1,'Ayush', 6)
    into t3 values (1,'Ayushi', 6)
    into t3 values (1,'A', 6)
    into t3 values (1,'Ay', 5
    into t3 values (1,'Ayu', 5)
    into t3 values (1,'Ayus', 3)
```

## Custom Datatypes
This Defines a data type email_address
```SQL
Create or replace TYPE email_address as object(
	username varchar(20),
	domain varchar(20)
)
```

I can use email_address in a table datatype

```SQL
create table emails(
user int,
email email_address
)
insert into emails values(1,email_address('aaa','google.com')) --aaa@google.com
```


## Multivalue Attributes

```SQL
create type phonelist as table of number(
	create table users(
	id number,
	name varchar(20),
	phone phone list,
	nested table phone store as phone_table,
	)
)
```
