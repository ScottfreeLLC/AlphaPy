################################################################################
#
# Package   : calendrical
# Created   : July 11, 2017
# Reference : Calendrical Calculations, Cambridge Press, 2002
#
# Copyright 2020 ScottFree Analytics LLC
# Mark Conway & Robert D. Scott II
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################


#
# Imports
#

import calendar
import logging
import math
import pandas as pd


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function expand_dates
#

def expand_dates(date_list):
    expanded_dates = []
    for item in date_list:
        if type(item) == str:
            expanded_dates.append(item)
        elif type(item) == list:
            start_date = item[0]
            end_date = item[1]
            dates_dt = pd.date_range(start_date, end_date).tolist()
            dates_str = [x.strftime('%Y-%m-%d') for x in dates_dt]
            expanded_dates.extend(dates_str)
        else:
            logger.info("Error in date: %s" % item)
    return expanded_dates


#
# Function biz_day_month
#

def biz_day_month(rdate):
    r"""Calculate the business day of the month.

    Parameters
    ----------
    rdate : int
        RDate date format.

    Returns
    -------
    bdm : int
        Business day of month.
    """

    gyear, gmonth, _ = rdate_to_gdate(rdate)
    rdate1 = gdate_to_rdate(gyear, gmonth, 1)

    bdm = 0
    index_date = rdate1
    while index_date <= rdate:
        dw = day_of_week(index_date)
        week_day = dw >= 1 and dw <= 5
        if week_day:
            bdm += 1
        index_date += 1

    holidays = set_holidays(gyear, True)
    for h in holidays:
        holiday = holidays[h]
        in_period = holiday >= rdate1 and holiday <= rdate
        dwh = day_of_week(holiday)
        week_day = dwh >= 1 and dwh <= 5
        if in_period and week_day:
            bdm -= 1
    return bdm


#
# Function biz_day_week
#

def biz_day_week(rdate):
    r"""Calculate the business day of the week.

    Parameters
    ----------
    rdate : int
        RDate date format.

    Returns
    -------
    bdw : int
        Business day of week.
    """

    gyear, _, _ = rdate_to_gdate(rdate)
    dw = day_of_week(rdate)
    week_day = dw >= 1 and dw <= 5

    bdw = 0
    if week_day:
        rdate1 = rdate - dw + 1
        rdate2 = rdate - 1
        holidays = set_holidays(gyear, True)
        for h in holidays:
            holiday = holidays[h]
            in_period = holiday >= rdate1 and holiday <= rdate2
            if in_period:
                bdw -= 1
    return bdw


#
# Function day_of_week
#

def day_of_week(rdate):
    r"""Get the ordinal day of the week.

    Parameters
    ----------
    rdate : int
        RDate date format.

    Returns
    -------
    dw : int
        Ordinal day of the week.
    """
    dw = rdate % 7
    return dw


#
# Function day_of_year
#

def day_of_year(gyear, gmonth, gday):
    r"""Calculate the day number of the given calendar year.

    Parameters
    ----------
    gyear : int
        Gregorian year.
    gmonth : int
        Gregorian month.
    gday : int
        Gregorian day.

    Returns
    -------
    dy : int
        Day number of year in RDate format.
    """
    dy = subtract_dates(gyear - 1, 12, 31, gyear, gmonth, gday)
    return dy


#
# Function days_left_in_year
#

def days_left_in_year(gyear, gmonth, gday):
    r"""Calculate the number of days remaining in the calendar year.

    Parameters
    ----------
    gyear : int
        Gregorian year.
    gmonth : int
        Gregorian month.
    gday : int
        Gregorian day.

    Returns
    -------
    days_left : int
        Calendar days remaining in RDate format.
    """
    days_left = subtract_dates(gyear, gmonth, gday, gyear, 12, 31)
    return days_left


#
# Function first_kday
#

def first_kday(k, gyear, gmonth, gday):
    r"""Calculate the first kday in RDate format.

    Parameters
    ----------
    k : int
        Day of the week.
    gyear : int
        Gregorian year.
    gmonth : int
        Gregorian month.
    gday : int
        Gregorian day.

    Returns
    -------
    fkd : int
        first-kday in RDate format.
    """
    fkd = nth_kday(1, k, gyear, gmonth, gday)
    return fkd


#
# Function gdate_to_rdate
#

def gdate_to_rdate(gyear, gmonth, gday):
    r"""Convert Gregorian date to RDate format.

    Parameters
    ----------
    gyear : int
        Gregorian year.
    gmonth : int
        Gregorian month.
    gday : int
        Gregorian day.

    Returns
    -------
    rdate : int
        RDate date format.
    """

    if gmonth <= 2:
        rfactor = 0
    elif gmonth > 2 and leap_year(gyear):
        rfactor = -1
    else:
        rfactor = -2

    rdate = 365 * (gyear - 1) \
            + math.floor((gyear - 1) / 4) \
            - math.floor((gyear - 1) / 100) \
            + math.floor((gyear - 1) / 400) \
            + math.floor(((367 * gmonth) - 362) / 12) \
            + gday + rfactor
    return(rdate)


#
# Function get_nth_kday_of_month
#
    
def get_nth_kday_of_month(gday, gmonth, gyear):
    r"""Convert Gregorian date to RDate format.

    Parameters
    ----------
    gday : int
        Gregorian day.
    gmonth : int
        Gregorian month.
    gyear : int
        Gregorian year.

    Returns
    -------
    nth : int
        Ordinal number of a given day's occurrence within the month,
        for example, the third Friday of the month.
    """

    this_month = calendar.monthcalendar(gyear, gmonth)
    nth_kday_tuple = next(((i, e.index(gday)) for i, e in enumerate(this_month) if gday in e), None)
    tuple_row = nth_kday_tuple[0]
    tuple_pos = nth_kday_tuple[1]
    nth = tuple_row + 1
    if tuple_row > 0 and this_month[0][tuple_pos] == 0:
        nth -= 1
    return nth


#
# Function get_rdate
#

def get_rdate(row):
    r"""Extract RDate from a dataframe.

    Parameters
    ----------
    row : pandas.DataFrame
        Row of a dataframe containing year, month, and day.

    Returns
    -------
    rdate : int
        RDate date format.
    """
    return gdate_to_rdate(row['year'], row['month'], row['day'])


#
# Function kday_after
#

def kday_after(rdate, k):
    r"""Calculate the day after a given RDate.

    Parameters
    ----------
    rdate : int
        RDate date format.
    k : int
        Day of the week.

    Returns
    -------
    kda : int
        kday-after in RDate format.
    """
    kda = kday_on_before(rdate + 7, k)
    return kda


#
# Function kday_before
#

def kday_before(rdate, k):
    r"""Calculate the day before a given RDate.

    Parameters
    ----------
    rdate : int
        RDate date format.
    k : int
        Day of the week.

    Returns
    -------
    kdb : int
        kday-before in RDate format.
    """
    kdb = kday_on_before(rdate - 1, k)
    return kdb


#
# Function kday_nearest
#

def kday_nearest(rdate, k):
    r"""Calculate the day nearest a given RDate.

    Parameters
    ----------
    rdate : int
        RDate date format.
    k : int
        Day of the week.

    Returns
    -------
    kdn : int
        kday-nearest in RDate format.
    """
    kdn = kday_on_before(rdate + 3, k)
    return kdn


#
# Function kday_on_after
#

def kday_on_after(rdate, k):
    r"""Calculate the day on or after a given RDate.

    Parameters
    ----------
    rdate : int
        RDate date format.
    k : int
        Day of the week.

    Returns
    -------
    kdoa : int
        kday-on-or-after in RDate format.
    """
    kdoa = kday_on_before(rdate + 6, k)
    return kdoa


#
# Function kday_on_before
#

def kday_on_before(rdate, k):
    r"""Calculate the day on or before a given RDate.

    Parameters
    ----------
    rdate : int
        RDate date format.
    k : int
        Day of the week.

    Returns
    -------
    kdob : int
        kday-on-or-before in RDate format.
    """
    kdob = rdate - day_of_week(rdate - k)
    return kdob


#
# Function last_kday
#

def last_kday(k, gyear, gmonth, gday):
    r"""Calculate the last kday in RDate format.

    Parameters
    ----------
    k : int
        Day of the week.
    gyear : int
        Gregorian year.
    gmonth : int
        Gregorian month.
    gday : int
        Gregorian day.

    Returns
    -------
    lkd : int
        last-kday in RDate format.
    """
    lkd = nth_kday(-1, k, gyear, gmonth, gday)
    return lkd


#
# Function leap_year
#

def leap_year(gyear):
    r"""Determine if this is a Gregorian leap year.

    Parameters
    ----------
    gyear : int
        Gregorian year.

    Returns
    -------
    leap_year : bool
        True if a Gregorian leap year, else False.
    """

    mod1 = (gyear % 4 == 0)
    mod2 = True
    if gyear % 100 == 0:
        mod2 = gyear % 400 == 0

    leap_year = False
    if mod1 and mod2:
        leap_year = True
    return leap_year


#
# Function next_event
#

def next_event(rdate, events):
    r"""Find the next event after a given date.

    Parameters
    ----------
    rdate : int
        RDate date format.
    events : list of RDate (int)
        Monthly events in RDate format.

    Returns
    -------
    event : RDate (int)
        Next event in RDate format.
    """
    try:
        event = next(e for e in events if e > rdate)
    except:
        event = 0
    return event


#
# Function next_holiday
#

def next_holiday(rdate, holidays):
    r"""Find the next holiday after a given date.

    Parameters
    ----------
    rdate : int
        RDate date format.
    holidays : dict of RDate (int)
        Holidays in RDate format.

    Returns
    -------
    holiday : RDate (int)
        Next holiday in RDate format.
    """
    try:
        holiday = next(h for h in sorted(holidays.values()) if h > rdate)
    except:
        holiday = 0
    return holiday


#
# Function nth_bizday
#

def nth_bizday(n, gyear, gmonth):
    r"""Calculate the nth business day in a month.

    Parameters
    ----------
    n : int
        Number of the business day to get.
    gyear : int
        Gregorian year.
    gmonth : int
        Gregorian month.

    Returns
    -------
    bizday : int
        Nth business day of a given month in RDate format.
    """

    rdate = gdate_to_rdate(gyear, gmonth, 1)
    holidays = set_holidays(gyear, True)
    ibd = 0
    idate = rdate
    while (ibd < n):
        dw = day_of_week(idate)
        week_day = dw >= 1 and dw <= 5
        if week_day and idate not in holidays.values():
            ibd += 1
            bizday = idate
        idate += 1
    return bizday


#
# Function nth_kday
#

def nth_kday(n, k, gyear, gmonth, gday):
    r"""Calculate the nth-kday in RDate format.

    Parameters
    ----------
    n : int
        Occurrence of a given day counting in either direction.
    k : int
        Day of the week.
    gyear : int
        Gregorian year.
    gmonth : int
        Gregorian month.
    gday : int
        Gregorian day.

    Returns
    -------
    nthkday : int
        nth-kday in RDate format.
    """

    rdate = gdate_to_rdate(gyear, gmonth, gday)
    if n > 0:
        nthkday = 7 * n + kday_before(rdate, k)
    else:
        nthkday = 7 * n + kday_after(rdate, k)
    return nthkday


#
# Function previous_event
#

def previous_event(rdate, events):
    r"""Find the previous event before a given date.

    Parameters
    ----------
    rdate : int
        RDate date format.
    events : list of RDate (int)
        Monthly events in RDate format.

    Returns
    -------
    event : RDate (int)
        Previous event in RDate format.
    """
    try:
        event = next(e for e in sorted(events, reverse=True) if e < rdate)
    except:
        event = 0
    return event


#
# Function previous_holiday
#

def previous_holiday(rdate, holidays):
    r"""Find the previous holiday before a given date.

    Parameters
    ----------
    rdate : int
        RDate date format.
    holidays : dict of RDate (int)
        Holidays in RDate format.

    Returns
    -------
    holiday : RDate (int)
        Previous holiday in RDate format.
    """
    try:
        holiday = next(h for h in sorted(holidays.values(), reverse=True) if h < rdate)
    except:
        holiday = 0
    return holiday


#
# Function rdate_to_gdate
#

def rdate_to_gdate(rdate):
    r"""Convert RDate format to Gregorian date format.

    Parameters
    ----------
    rdate : int
        RDate date format.

    Returns
    -------
    gyear : int
        Gregorian year.
    gmonth : int
        Gregorian month.
    gday : int
        Gregorian day.
    """

    gyear = rdate_to_gyear(rdate)
    priordays = rdate - gdate_to_rdate(gyear, 1, 1)
    value1 = gdate_to_rdate(gyear, 3, 1)
    if rdate < value1:
        correction = 0
    elif rdate >= value1 and leap_year(gyear):
        correction = 1
    else:
        correction = 2
    gmonth = math.floor((12 * (priordays + correction) + 373) / 367)
    gday = rdate - gdate_to_rdate(gyear, gmonth, 1) + 1
    return gyear, gmonth, gday


#
# Function rdate_to_gyear
#

def rdate_to_gyear(rdate):
    r"""Convert RDate format to Gregorian year.

    Parameters
    ----------
    rdate : int
        RDate date format.

    Returns
    -------
    gyear : int
        Gregorian year.
    """

    d0 = rdate - 1
    n400 = math.floor(d0 / 146097)
    d1 = d0 % 146097
    n100 = math.floor(d1 / 36524)
    d2 = d1 % 36524
    n4 = math.floor(d2 / 1461)
    d3 = d2 % 1461
    n1 = math.floor(d3 / 365)

    theyear = 400 * n400 + 100 * n100 + 4 * n4 + n1
    if n100 == 4 or n1 == 4:
        gyear = theyear
    else:
        gyear = theyear + 1
    return gyear


#
# Function set_events
#

def set_events(n, k, gyear, gday):
    r"""Define monthly events for a given year.

    Parameters
    ----------
    n : int
        Occurrence of a given day counting in either direction.
    k : int
        Day of the week.
    gyear : int
        Gregorian year for the events.
    gday : int
        Gregorian day representing the first day to consider.

    Returns
    -------
    events : list of RDate (int)
        Monthly events in RDate format.

    Example
    -------
    >>> # Options Expiration (Third Friday of every month)
    >>> set_events(3, 5, 2017, 1)
    """

    events = []
    month_range = range(1, 13)
    for m in month_range:
        rdate = nth_kday(n, k, gyear, m, gday)
        events.append(rdate)
    return events


#
# Function subtract_dates
#

def subtract_dates(gyear1, gmonth1, gday1, gyear2, gmonth2, gday2):
    r"""Calculate the difference between two Gregorian dates.

    Parameters
    ----------
    gyear1 : int
        Gregorian year of first date.
    gmonth1 : int
        Gregorian month of first date.
    gday1 : int
        Gregorian day of first date.
    gyear2 : int
        Gregorian year of successive date.
    gmonth2 : int
        Gregorian month of successive date.
    gday2 : int
        Gregorian day of successive date.

    Returns
    -------
    delta_days : int
        Difference in days in RDate format.
    """
    delta_days = gdate_to_rdate(gyear2, gmonth2, gday2) \
                 - gdate_to_rdate(gyear1, gmonth1, gday1)
    return delta_days



#
# Holiday Functions in Calendar Order
#


#
# Function new_years_day
#

def new_years_day(gyear, observed):
    r"""Get New Year's day for a given year.

    Parameters
    ----------
    gyear : int
        Gregorian year.
    observed : bool
        False if the exact date, True if the weekday.

    Returns
    -------
    nyday : int
        New Year's Day in RDate format.
    """
    nyday = gdate_to_rdate(gyear, 1, 1)
    if observed and day_of_week(nyday) == 0:
        nyday += 1
    return nyday


#
# Function mlk_day
#

def mlk_day(gyear):
    r"""Get Martin Luther King Day for a given year.

    Parameters
    ----------
    gyear : int
        Gregorian year.

    Returns
    -------
    mlkday : int
        Martin Luther King Day in RDate format.
    """
    mlkday = nth_kday(3, 1, gyear, 1, 1)
    return mlkday


#
# Function valentines_day
#

def valentines_day(gyear):
    r"""Get Valentine's day for a given year.

    Parameters
    ----------
    gyear : int
        Gregorian year.

    Returns
    -------
    valentines : int
        Valentine's Day in RDate format.
    """
    valentines = gdate_to_rdate(gyear, 2, 14)
    return valentines


#
# Function presidents_day
#

def presidents_day(gyear):
    r"""Get President's Day for a given year.

    Parameters
    ----------
    gyear : int
        Gregorian year.

    Returns
    -------
    prezday : int
        President's Day in RDate format.
    """
    prezday = nth_kday(3, 1, gyear, 2, 1)
    return prezday


#
# Function saint_patricks_day
#

def saint_patricks_day(gyear):
    r"""Get Saint Patrick's day for a given year.

    Parameters
    ----------
    gyear : int
        Gregorian year.
    observed : bool
        False if the exact date, True if the weekday.

    Returns
    -------
    patricks : int
        Saint Patrick's Day in RDate format.
    """
    patricks = gdate_to_rdate(gyear, 3, 17)
    return patricks


#
# Function good_friday
#

def good_friday(gyear):
    r"""Get Good Friday for a given year.

    Parameters
    ----------
    gyear : int
        Gregorian year.

    Returns
    -------
    gf : int
        Good Friday in RDate format.
    """
    gf = easter_day(gyear) - 2
    return gf


#
# Function easter_day
#

def easter_day(gyear):
    r"""Get Easter Day for a given year.

    Parameters
    ----------
    gyear : int
        Gregorian year.

    Returns
    -------
    ed : int
        Easter Day in RDate format.
    """

    century = math.floor(gyear / 100) + 1
    epacts = (14 + 11 * (gyear % 19) - math.floor(3 * century / 4) \
             + math.floor((5 + 8 * century) / 25)) % 30
    if epacts == 0 or (epacts == 1 and 10 < (gyear % 19)):
        epacta = epacts + 1
    else:
        epacta = epacts
    rdate = gdate_to_rdate(gyear, 4, 19) - epacta
    ed = kday_after(rdate, 0)
    return ed


#
# Function cinco_de_mayo
#

def cinco_de_mayo(gyear):
    r"""Get Cinco de Mayo for a given year.

    Parameters
    ----------
    gyear : int
        Gregorian year.

    Returns
    -------
    cinco_de_mayo : int
        Cinco de Mayo in RDate format.
    """
    cinco = gdate_to_rdate(gyear, 5, 5)
    return cinco


#
# Function mothers_day
#

def mothers_day(gyear):
    r"""Get Mother's Day for a given year.

    Parameters
    ----------
    gyear : int
        Gregorian year.

    Returns
    -------
    mothers_day : int
        Mother's Day in RDate format.
    """
    mothers_day = nth_kday(2, 0, gyear, 5, 1)
    return mothers_day


#
# Function memorial_day
#

def memorial_day(gyear):
    r"""Get Memorial Day for a given year.

    Parameters
    ----------
    gyear : int
        Gregorian year.

    Returns
    -------
    md : int
        Memorial Day in RDate format.
    """
    md = last_kday(1, gyear, 5, 31)
    return md


#
# Function fathers_day
#

def fathers_day(gyear):
    r"""Get Father's Day for a given year.

    Parameters
    ----------
    gyear : int
        Gregorian year.

    Returns
    -------
    fathers_day : int
        Father's Day in RDate format.
    """
    fathers_day = nth_kday(3, 0, gyear, 6, 1)
    return fathers_day


#
# Function independence_day
#

def independence_day(gyear, observed):
    r"""Get Independence Day for a given year.

    Parameters
    ----------
    gyear : int
        Gregorian year.
    observed : bool
        False if the exact date, True if the weekday.

    Returns
    -------
    d4j : int
        Independence Day in RDate format.
    """
    d4j = gdate_to_rdate(gyear, 7, 4)
    if observed:
        if day_of_week(d4j) == 6:
            d4j -= 1
        if day_of_week(d4j) == 0:
            d4j += 1
    return d4j


#
# Function labor_day
#

def labor_day(gyear):
    r"""Get Labor Day for a given year.

    Parameters
    ----------
    gyear : int
        Gregorian year.

    Returns
    -------
    lday : int
        Labor Day in RDate format.
    """
    lday = first_kday(1, gyear, 9, 1)
    return lday


#
# Function halloween
#

def halloween(gyear):
    r"""Get Halloween for a given year.

    Parameters
    ----------
    gyear : int
        Gregorian year.

    Returns
    -------
    halloween : int
        Halloween in RDate format.
    """
    halloween = gdate_to_rdate(gyear, 10, 31)
    return halloween


#
# Function veterans_day
#

def veterans_day(gyear, observed):
    r"""Get Veteran's day for a given year.

    Parameters
    ----------
    gyear : int
        Gregorian year.
    observed : bool
        False if the exact date, True if the weekday.

    Returns
    -------
    veterans : int
        Veteran's Day in RDate format.
    """
    veterans = gdate_to_rdate(gyear, 11, 11)
    if observed and day_of_week(veterans) == 0:
        veterans += 1
    return veterans


#
# Function thanksgiving_day
#

def thanksgiving_day(gyear):
    r"""Get Thanksgiving Day for a given year.

    Parameters
    ----------
    gyear : int
        Gregorian year.

    Returns
    -------
    tday : int
        Thanksgiving Day in RDate format.
    """
    tday = nth_kday(4, 4, gyear, 11, 1)
    return tday


#
# Function christmas_day
#

def christmas_day(gyear, observed):
    r"""Get Christmas Day for a given year.

    Parameters
    ----------
    gyear : int
        Gregorian year.
    observed : bool
        False if the exact date, True if the weekday.

    Returns
    -------
    xmas : int
        Christmas Day in RDate format.
    """
    xmas = gdate_to_rdate(gyear, 12, 25)
    if observed:
        if day_of_week(xmas) == 6:
            xmas -= 1
        if day_of_week(xmas) == 0:
            xmas += 1
    return xmas


#
# Define holiday map
#

holiday_map = {"New Year's Day"    : (new_years_day, True),
               "MLK Day"           : (mlk_day, False),
               "Valentine's Day"   : (valentines_day, False),
               "President's Day"   : (presidents_day, False),
               "St. Patrick's Day" : (saint_patricks_day, False),
               "Good Friday"       : (good_friday, False),
               "Easter"            : (easter_day, False),
               "Cinco de Mayo"     : (cinco_de_mayo, False),
               "Mother's Day"      : (mothers_day, False),
               "Memorial Day"      : (memorial_day, False),
               "Father's Day"      : (fathers_day, False),
               "Independence Day"  : (independence_day, True),
               "Labor Day"         : (labor_day, False),
               "Halloween"         : (halloween, False),
               "Veteran's Day"     : (veterans_day, True),
               "Thanksgiving"      : (thanksgiving_day, False),
               "Christmas"         : (christmas_day, True)}


#
# Function get_holiday_names
#

def get_holiday_names():
    r"""Get the list of defined holidays.

    Returns
    -------
    holidays : list of str
        List of holiday names.
    """
    holidays = [h for h in holiday_map]
    return holidays


#
# Function set_holidays
#

def set_holidays(gyear, observe):
    r"""Determine if this is a Gregorian leap year.

    Parameters
    ----------
    gyear : int
        Value for the corresponding key.
    observe : bool
        True to get the observed date, otherwise False.

    Returns
    -------
    holidays : dict of int
        Set of holidays in RDate format for a given year.
    """

    holidays = {}
    for h in holiday_map:
        hfunc = holiday_map[h][0]
        observed = holiday_map[h][1]
        if observed:
            holidays[h] = hfunc(gyear, observe)
        else:
            holidays[h] = hfunc(gyear)
    return holidays
