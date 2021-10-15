
-- Number of rows per table
select count(*) from booking; 	# 198 998
select count(*) from user;		# 166 477	
select count(*) from charter;	#   6 348
select count(*) from package;	#  24 363

-- Tables preview
select * from booking limit 5;
select * from user limit 5;
select * from charter limit 5;
select * from package limit 5;

-- Check foreign keys
select count(*) from package where charter_id is null; 	# Every package has charter
select count(*) from booking where (charter_id is null) or (package_id is null) or (user_id is null); # Every reservation is complete

-- Users without reservation --> only 3 users
select count(*)
from user
	left join booking on booking.user_id = user.id
where booking.user_id is null;

-- Reservation without user details	--> 370 
select count(*)
from booking
	left join user on booking.user_id = user.id
where user.id is null;

-- Packages without reservation --> All packages have been used
select count(*)
from package
	left join booking on booking.package_id = package.id
where booking.package_id is null;

-- Reservation without package details --> 0
select count(*)
from booking
	left join package on booking.package_id = package.id
where package.id is null;

-- Charter without reservation --> All charters have been used
select count(*)
from charter
	left join booking on booking.charter_id = charter.id
where booking.charter_id is null;

-- Reservations without charter details --> 6074 (charter table miss these information)
select count(*)
from booking
	left join charter on booking.charter_id = charter.id
where charter.id is null;

-- All reservations with complete details about users, packages and charts
select count(*)
from booking
	inner join user on booking.user_id = user.id
    inner join package on booking.package_id = package.id
    inner join charter on booking.charter_id = charter.id;
    
-- Return relevant features for all reservations with complete details about users, packages and charts
select b.id, b.charter_id, b.package_id, b.user_id, b.status, b.trip_date, b.person_count, b.children_count, b.instantly_booked, b.date_created,
u.city as 'user_city', u.state as 'user_state', u.country as 'user_country',
ch.location, ch.state, ch.country, ch.captain_id, ch.public, ch.title as 'charter_title', ch.anglers_choice_award, 
ch.lake_fishing, ch.river_fishing, ch.inshore_fishing, ch.offshore_fishing, ch.big_game_fishing, ch.bottom_fishing, ch.trolling, ch.light_tackle, ch.heavy_tackle, ch.fly_fishing, ch.jigging, ch.lunch_included, ch.snacks_included, ch.drinks_included,
p.title as 'package_title', p.price, p.shared, p.seasonal, p.duration_hours, p.departure_time
from booking b
	inner join user u on b.user_id = u.id
    inner join package p on b.package_id = p.id
    inner join charter ch on b.charter_id = ch.id
where (p.title != '' and p.title is not null) 
	and (ch.location != '' and ch.location is not null)
    and (ch.country != '' and ch.country is not null)
    and (u.city != '' and u.city is not null)
    and (u.country != '' and u.country is not null)
limit 20000;

