% 1. Facts
% male
male(charles).
male(william).
male(harry).
male(george).
male(louis).
male(archie).

% female
female(diana).
female(camila).
female(catherine).
female(meghan).
female(charlotte).

% spouse
spouse(charles,diana).
spouse(charles,camila).
spouse(william,catherine).
spouse(harry,meghan).
spouse(diana,charles).
spouse(camila,charles).
spouse(catherine,william).
spouse(meghan,harry).

% parent
parent(charles,william).
parent(charles,harry).
parent(diana,william).
parent(diana,harry).
parent(harry,archie).
parent(meghan,archie).
parent(william,george).
parent(william,charlotte).
parent(william,louis).
parent(catherine,george).
parent(catherine,charlotte).
parent(catherine,louis).

% 2. Rules
mother(X,Y) :- parent(X,Y), female(X).
father(X,Y) :- parent(X,Y), male(X).
stepmother(X,Y) :- female(X), parent(A,Y), spouse(A,X), mother(B,Y), not(X = B).
brother(X,Y) :- sibling(X,Y), male(X).
sister(X,Y) :- sibling(X,Y), female(X).
sibling(X,Y) :- parent(A,X), parent(A,Y), not(X = Y).
nephew(X,Y) :- sibling(Y,A), parent(A,X), male(X).
niece(X,Y) :- sibling(Y,A), parent(A,X), female(X).
grandparent(X,Y) :- parent(X,A), parent(A,Y).
cousin(X,Y) :- parent(A,X), parent(B,Y), sibling(A,B).
uncle(X,Y) :- male(X), nephew(Y,X).
uncle(X,Y) :- male(X), niece(Y,X).
aunt(X,Y) :- female(X), nephew(Y,X).
aunt(X,Y) :- female(X), niece(Y,X).
aunt(X,Y) :- female(X), spouse(X,A), uncle(A,Y).
decendant(X,Y) :- parent(X,Y).
decendant(X,Y) :- grandparent(X,Y).
