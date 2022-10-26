/**
 * SyoTimer - countdown jquery plugin
 * @version: 2.0.0 
 * @author: John Syomochkin 
 * @homepage: https://github.com/mrfratello/SyoTimer#readme 
 * @date: 2017.6.24
 * @license: under MIT license
 */
(function($){
    var DAY = "day",
        HOUR = "hour",
        MINUTE = "minute",
        SECOND = "second";
    var DAY_IN_SEC = 24 * 60 * 60;
    var HOUR_IN_SEC = 60 * 60;
    var MINUTE_IN_SEC = 60;
    var LAYOUT_TYPES = {
            d: DAY,
            h: HOUR,
            m: MINUTE,
            s: SECOND
        };
    var UNIT_LINKED_LIST = {
            list: [SECOND, MINUTE, HOUR, DAY],
            next: function(current) {
                var currentIndex = this.list.indexOf(current);
                return (currentIndex < this.list.length ) ? this.list[currentIndex + 1] : false;
            },
            prev: function(current) {
                var currentIndex = this.list.indexOf(current);
                return (currentIndex > 0 ) ? this.list[currentIndex - 1] : false;
            }
        };

    var DEFAULTS = {
        year: 2014,
        month: 7,
        day: 31,
        hour: 0,
        minute: 0,
        second: 0,
        timeZone: 'local',          // setting the time zone of deadline.
                                    // If 'local' then the time zone is ignored and
                                    // the deadline is determined by local time of the user.
                                    // Otherwise, specifies the offset from the UTC
        ignoreTransferTime: false,  // If `true` then transfer to summer/winter time will not be considered.
        layout: 'dhms',             // sets an order of layout of units of the timer:
                                    // days (d) of hours ('h'), minute ('m'), second ('s').
        periodic: false,            //`true` - the timer is periodic.
                                    // If the date until which counts the timer is reached,
                                    // the next value date which will count down
                                    // the timer is incremented by the value `periodInterval`
        periodInterval: 7,          // the period of the timer in `periodUnit`
                                    // (if `periodic` is set to `true`)
        periodUnit: 'd',            // the unit of measurement period timer

        doubleNumbers: true,        // `true` - show hours, minutes and seconds with leading zeros
                                    // (2 hours 5 minutes 4 seconds = 02:05:04)
        effectType: 'none',         // The effect of changing the value of seconds
        lang: 'eng',                // localization of a countdown signatures (days, hours, minutes, seconds)
        headTitle: '',              // text above the countdown (may be as html string)
        footTitle: '',              // text under the countdown (may be as html string)
        afterDeadline: function(timerBlock){
            timerBlock.bodyBlock.html('<p style="font-size: 1.2em;">The countdown is finished!</p>');
        }
    };

    var ITEMS_HAS_OPTIONS = {
        second: false,
        minute: false,
        hour: false,
        day: false
    };

    var SyoTimer = {
        /**
         * Init syotimer on DOM
         * @param settings
         * @returns {Array|Object|*}
         */
        init: function(settings) {
            var options = $.extend({}, DEFAULTS, settings || {});
            options.itemTypes = staticMethod.getItemTypesByLayout(options.layout);
            options._itemsHas = $.extend({}, ITEMS_HAS_OPTIONS);
            for (var i = 0; i < options.itemTypes.length; i++) {
                options._itemsHas[options.itemTypes[i]] = true;
            }
            return this.each(function() {
                var elementBox = $(this);
                elementBox.data('syotimer-options', options);
                SyoTimer._render.apply(this, []);
                SyoTimer._perSecondHandler.apply(this, []);
            });
        },

        /**
         * Rendering base elements of countdown
         * @private
         */
        _render: function() {
            var elementBox = $(this),
                options = elementBox.data('syotimer-options');

            var timerItem = staticMethod.getTimerItem(),
                headBlock = $('<div/>', {"class": 'syotimer__head'})
                    .html(options.headTitle),
                bodyBlock = $('<div/>', {"class": 'syotimer__body'}),
                footBlock = $('<div/>', {"class": 'syotimer__footer'})
                    .html(options.footTitle),
                itemBlocks = {};

            for (var i = 0; i < options.itemTypes.length; i++) {
                var item = timerItem.clone();
                item.addClass('syotimer-cell_type_' + options.itemTypes[i]);
                bodyBlock.append(item);
                itemBlocks[options.itemTypes[i]] = item;
            }
            var timerBlocks = {
                    headBlock: headBlock,
                    bodyBlock: bodyBlock,
                    footBlock: footBlock
                };
            elementBox.data('syotimer-blocks', timerBlocks)
                .data('syotimer-items', itemBlocks)
                .addClass('syotimer')
                .append(headBlock)
                .append(bodyBlock)
                .append(footBlock);
        },

        /**
         * Handler called per seconds while countdown is not over
         * @private
         */
        _perSecondHandler: function() {
            var elementBox = $(this),
                options = elementBox.data('syotimer-options');
            $('.syotimer-cell > .syotimer-cell__value', elementBox).css( 'opacity', 1 );
            var currentDate = new Date(),
                deadLineDate = new Date(
                    options.year,
                    options.month - 1,
                    options.day,
                    options.hour,
                    options.minute,
                    options.second
                ),
                differenceInMilliSec = staticMethod.getDifferenceWithTimezone(currentDate, deadLineDate, options),
                secondsToDeadLine = staticMethod.getSecondsToDeadLine(differenceInMilliSec, options);
            if ( secondsToDeadLine >= 0 ) {
                SyoTimer._refreshUnitsDom.apply(this, [secondsToDeadLine]);
                SyoTimer._applyEffectSwitch.apply(this, [options.effectType]);
            } else {
                elementBox = $.extend(elementBox, elementBox.data('syotimer-blocks'));
                options.afterDeadline( elementBox );
            }
        },

        /**
         * Refresh unit DOM of countdown
         * @param secondsToDeadLine
         * @private
         */
        _refreshUnitsDom: function(secondsToDeadLine) {
            var elementBox = $(this),
                options = elementBox.data('syotimer-options'),
                itemBlocks = elementBox.data('syotimer-items'),
                unitList = options.itemTypes,
                unitsToDeadLine = staticMethod.getUnitsToDeadLine( secondsToDeadLine );

            if ( !options._itemsHas.day ) {
                unitsToDeadLine.hour += unitsToDeadLine.day * 24;
            }
            if ( !options._itemsHas.hour ) {
                unitsToDeadLine.minute += unitsToDeadLine.hour * 60;
            }
            if ( !options._itemsHas.minute ) {
                unitsToDeadLine.second += unitsToDeadLine.minute * 60;
            }
            for(var i = 0; i < unitList.length; i++) {
                var unit = unitList[i],
                    unitValue = unitsToDeadLine[unit],
                    itemBlock = itemBlocks[unit];
                itemBlock.data('syotimer-unit-value', unitValue);
                $('.syotimer-cell__value', itemBlock).html(staticMethod.format2(
                    unitValue,
                    (unit !== DAY) ? options.doubleNumbers : false
                ));
                $('.syotimer-cell__unit', itemBlock).html($.syotimerLang.getNumeral(
                    unitValue,
                    options.lang,
                    unit
                ));
            }
        },

        /**
         * Applying effect of changing numbers
         * @param effectType
         * @param unit
         * @private
         */
        _applyEffectSwitch: function(effectType, unit) {
            unit = unit || SECOND;
            var element = this,
                elementBox = $(element);
            if ( effectType === 'none' ) {
                setTimeout(function () {
                    SyoTimer._perSecondHandler.apply(element, []);
                }, 1000);
            } else if ( effectType === 'opacity' ) {
                var itemBlocks = elementBox.data('syotimer-items'),
                    unitItemBlock = itemBlocks[unit];
                if (unitItemBlock) {
                    var nextUnit = UNIT_LINKED_LIST.next(unit),
                        unitValue = unitItemBlock.data('syotimer-unit-value');
                    $('.syotimer-cell__value', unitItemBlock).animate(
                        {opacity: 0.1},
                        1000,
                        'linear',
                        function () {
                            SyoTimer._perSecondHandler.apply(element, []);
                        }
                    );
                    if (nextUnit && unitValue === 0) {
                        SyoTimer._applyEffectSwitch.apply(element, [effectType, nextUnit]);
                    }
                }
            }
        }
    };

    var staticMethod = {
        /**
         * Return once cell DOM of countdown: day, hour, minute, second
         * @returns {object}
         */
        getTimerItem: function() {
            var timerCellValue = $('<div/>', {
                    "class": 'syotimer-cell__value',
                    "text": '0'
                }),
                timerCellUnit = $('<div/>', {"class": 'syotimer-cell__unit'}),
                timerCell = $('<div/>', {"class": 'syotimer-cell'});
            timerCell.append(timerCellValue)
                .append(timerCellUnit);
            return timerCell;
        },

        getItemTypesByLayout: function(layout) {
            var itemTypes = [];
            for (var i = 0; i < layout.length; i++) {
                itemTypes.push(LAYOUT_TYPES[layout[i]]);
            }
            return itemTypes;
        },

        /**
         * Getting count of seconds to deadline
         * @param differenceInMilliSec
         * @param options
         * @returns {*}
         */
        getSecondsToDeadLine: function(differenceInMilliSec, options) {
            var secondsToDeadLine,
                differenceInSeconds = differenceInMilliSec / 1000;
            differenceInSeconds = Math.floor( differenceInSeconds );
            if ( options.periodic ) {
                var additionalInUnit,
                    differenceInUnit,
                    periodUnitInSeconds = staticMethod.getPeriodUnit(options.periodUnit),
                    fullTimeUnitsBetween = differenceInMilliSec / (periodUnitInSeconds * 1000);
                fullTimeUnitsBetween = Math.ceil( fullTimeUnitsBetween );
                fullTimeUnitsBetween = Math.abs( fullTimeUnitsBetween );
                if ( differenceInSeconds >= 0 ) {
                    differenceInUnit = fullTimeUnitsBetween % options.periodInterval;
                    differenceInUnit = ( differenceInUnit === 0 )? options.periodInterval : differenceInUnit;
                    differenceInUnit -= 1;
                } else {
                    differenceInUnit = options.periodInterval - fullTimeUnitsBetween % options.periodInterval;
                }
                additionalInUnit = differenceInSeconds % periodUnitInSeconds;

                // fix когда дедлайн раньше текущей даты,
                // возникает баг с неправильным расчетом интервала при different пропорциональной periodUnit
                if ( ( additionalInUnit === 0 ) && ( differenceInSeconds < 0 ) ) {
                    differenceInUnit--;
                }
                secondsToDeadLine = Math.abs( differenceInUnit * periodUnitInSeconds + additionalInUnit );
            } else {
                secondsToDeadLine = differenceInSeconds;
            }
            return secondsToDeadLine;
        },

        /**
         * Getting count of units to deadline
         * @param secondsToDeadLine
         * @returns {{}}
         */
        getUnitsToDeadLine: function(secondsToDeadLine) {
            var unit = DAY,
                unitsToDeadLine = {};
            do {
                var unitInMilliSec = staticMethod.getPeriodUnit(unit);
                unitsToDeadLine[unit] = Math.floor(secondsToDeadLine / unitInMilliSec);
                secondsToDeadLine = secondsToDeadLine % unitInMilliSec;
            } while (unit = UNIT_LINKED_LIST.prev(unit));
            return unitsToDeadLine;
        },

        /**
         * Determine a unit of period in milliseconds
         * @param given_period_unit
         * @returns {number}
         */
        getPeriodUnit: function(given_period_unit) {
            switch (given_period_unit) {
                case 'd':
                case DAY:
                    return DAY_IN_SEC;
                case 'h':
                case HOUR:
                    return HOUR_IN_SEC;
                case 'm':
                case MINUTE:
                    return MINUTE_IN_SEC;
                case 's':
                case SECOND:
                    return 1;
            }
        },

        getDifferenceWithTimezone: function(currentDate, deadLineDate, options) {
            var differenceByLocalTimezone = deadLineDate.getTime() - currentDate.getTime(),
                amendmentOnTimezone = 0,
                amendmentOnTransferTime = 0,
                amendment;
            if ( options.timeZone !== 'local' ) {
                var timezoneOffset = parseFloat(options.timeZone) * staticMethod.getPeriodUnit(HOUR),
                    localTimezoneOffset = - currentDate.getTimezoneOffset() * staticMethod.getPeriodUnit(MINUTE);
                amendmentOnTimezone = (timezoneOffset - localTimezoneOffset) * 1000;
            }
            if ( options.ignoreTransferTime ) {
                var currentTimezoneOffset = -currentDate.getTimezoneOffset() * staticMethod.getPeriodUnit(MINUTE),
                    deadLineTimezoneOffset = -deadLineDate.getTimezoneOffset() * staticMethod.getPeriodUnit(MINUTE);
                amendmentOnTransferTime = (currentTimezoneOffset - deadLineTimezoneOffset) * 1000;
            }
            amendment = amendmentOnTimezone + amendmentOnTransferTime;
            return differenceByLocalTimezone - amendment;
        },

        /**
         * Formation of numbers with leading zeros
         * @param number
         * @param isUse
         * @returns {string}
         */
        format2: function(number, isUse) {
            isUse = (isUse !== false);
            return ( ( number <= 9 ) && isUse ) ? ( "0" + number ) : ( "" + number );
        }
    };

    var methods = {
        setOption: function(name, value) {
            var elementBox = $(this),
                options = elementBox.data('syotimer-options');
            if ( options.hasOwnProperty( name ) ) {
                options[name] = value;
                elementBox.data('syotimer-options', options);
            }
        }
    };

    $.fn.syotimer = function(options){
        if ( typeof options === 'string' && ( options === "setOption" ) ) {
            var otherArgs = Array.prototype.slice.call(arguments, 1);
            return this.each(function() {
                methods[options].apply( this, otherArgs );
            });
        } else if (options === null || typeof options === 'object'){
            return SyoTimer.init.apply(this, [options]);
        } else {
            $.error('SyoTimer. Error in call methods: methods is not exist');
        }
    };

    $.syotimerLang = {
        rus: {
            second: ['секунда', 'секунды', 'секунд'],
            minute: ['минута', 'минуты', 'минут'],
            hour: ['час', 'часа', 'часов'],
            day: ['день', 'дня', 'дней'],
            handler: 'rusNumeral'
        },
        eng: {
            second: ['second', 'seconds'],
            minute: ['minute', 'minutes'],
            hour: ['hour', 'hours'],
            day: ['day', 'days']
        },
        por: {
            second: ['segundo', 'segundos'],
            minute: ['minuto', 'minutos'],
            hour: ['hora', 'horas'],
            day: ['dia', 'dias']
        },
        spa: {
            second: ['segundo', 'segundos'],
            minute: ['minuto', 'minutos'],
            hour: ['hora', 'horas'],
            day: ['día', 'días']
        },
        heb: {
            second: ['שניה', 'שניות'],
            minute: ['דקה', 'דקות'],
            hour: ['שעה', 'שעות'],
            day: ['יום', 'ימים']
        },

        /**
         * Universal function for get correct inducement of nouns after a numeral (`number`)
         * @param number
         * @returns {number}
         */
        universal: function(number) {
            return ( number === 1 ) ? 0 : 1;
        },

        /**
         * Get correct inducement of nouns after a numeral for Russian language (rus)
         * @param number
         * @returns {number}
         */
        rusNumeral: function(number) {
            var cases = [2, 0, 1, 1, 1, 2],
                index;
            if ( number % 100 > 4 && number % 100 < 20 ) {
                index = 2;
            } else {
                index = cases[(number % 10 < 5) ? number % 10 : 5];
            }
            return index;
        },

        /**
         * Getting the correct declension of words after numerals
         * @param number
         * @param lang
         * @param unit
         * @returns {string}
         */
        getNumeral: function(number, lang, unit) {
            var handlerName = $.syotimerLang[lang].handler || 'universal',
                index = this[handlerName](number);
            return $.syotimerLang[lang][unit][index];
        }
    };
})(jQuery);
