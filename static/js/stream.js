var request_interval = 500;
var last_request = 0;
var isPaused = false;
var markerThreshold = 0.1;
var markerTimeout = 1750;
var markers = {};
var cardThreshold = 0.1;

//////////////////////////  VISUALIZATION  ////////////////////////////
var cnames = {

    'European Starling': 'Star',
    'Common House-Martin': 'Mehlschwalbe',
    'Eurasian Linnet': 'Bluthänfling',
    'European Pied Flycatcher': 'Trauerschnäpper',
    'Goldcrest': 'Wintergoldhähnchen',
    'Yellowhammer': 'Goldammer',
    'Eurasian Blue Tit': 'Blaumeise',
    'Common Chiffchaff': 'Zilpzalp',
    'European Goldfinch': 'Stieglitz',
    'Great Tit': 'Kohlmeise',
    'House Sparrow': 'Hausperling',
    'European Robin': 'Rotkelchen',
    'Black Redstart': 'Hausrotschwanz',
    'Common Chaffinch': 'Buchfink',
    'Eurasian Blackbird': 'Amsel'
}

function showFileAnalysis(analysis) {

    // Get time
    var d = new Date();
    var now = d.getTime();

    // Get canvas and context
    var canvas = document.getElementById('spec');
    var ctx = canvas.getContext("2d");

    // Get canvas position
    var canvasTRX = $('#spec').position().left + $('#spec').width();    
    var canvasTRY = 0; //$('#spec').position().top;
    var canvasHeight = $('#spec').height();
    var icon_size = canvasHeight * 0.25;

    // Hide current detection cards
    $('#d1').addClass('d-none');
    $('#d2').addClass('d-none');

    // Parse analysis and show marker
    var a_keys = Object.keys(analysis.prediction[0]);
    var cnt = 1;
    for (var k in a_keys) {

        // Get score fo current prediction
        var score = parseFloat(analysis.prediction[0][k].score);

        // Get time of last marker
        var lastMarker = 0;
        if (analysis.prediction[0][k].species in markers) {

            lastMarker = markers[analysis.prediction[0][k].species];

        }        

        if (score >= markerThreshold && now - lastMarker >= markerTimeout) {

            // Log
            //console.log('Drawing marker for ' + analysis.prediction[0][k].species + ' with score ' + score);

            // Draw line
            ctx.strokeStyle = "#FFFFFF";
            ctx.lineWidth = canvasHeight * 0.015;
            ctx.beginPath(); 
            ctx.moveTo(canvasTRX - (icon_size * 0.5), canvasTRY + icon_size);
            ctx.lineTo(canvasTRX - (icon_size * 0.5), canvasTRY + icon_size + canvasHeight);
            ctx.stroke();

            // Draw icon            
            var icon = new Image();
            icon.onload = function() {
                ctx.drawImage(icon, canvasTRX - icon_size, canvasTRY + (icon_size * 0.1), icon_size, icon_size);
            };
            icon.src = "static/img/" + analysis.prediction[0][k].species + ".jpg";

            // Set time
            markers[analysis.prediction[0][k].species] = now;

        }

        // Show detection card
        if (score >= cardThreshold) {

            var sname = analysis.prediction[0][k].species.split('_')[0]
            var cname = analysis.prediction[0][k].species.split('_')[1]

            $('#d' + cnt).removeClass('d-none');
            $('#d' + cnt + "-header").html("<b>" + cnames[cname] + "</b> (<i>" + sname + "</i>)");
            $('#d' + cnt + "-img").attr('src', "static/img/" + analysis.prediction[0][k].species + ".jpg");
            $('#d' + cnt + "-score").text("" + parseInt(score * 100) + "%");

        }

        cnt += 1;
        if (cnt > 2) break;

    }

    // Keep interval    
    if (now - last_request < request_interval)  setTimeout('requestStreamAnalysis()', request_interval - (now - last_request));

}

////////////////////////////  REQUEST  //////////////////////////////
function pauseRequests() {

    isPaused = true;

}

function resumeRequests() {

    isPaused = false;

}

function restartRequests() {

    var d = new Date();
    var now = d.getTime();

    if (now - last_request > request_interval * 2 && !isPaused) {

        console.log('Restarting requests...');
        requestStreamAnalysis();
    }

}

function requestStreamAnalysis() {

    // Do nothing if paused
    if (isPaused) return;

    // Set time of this request
    var d = new Date();
    last_request = d.getTime();

    // Prepare payload
    var json_array = {

        action: 'analysis'

    };
    json_string = JSON.stringify(json_array);

    // Make request
    $.ajax({
        url: 'process',
        type: 'POST',
        data: typeof json_string === "string" ? "json=" + encodeURIComponent(json_string) : json_string,
        async: true,
        success: function (response) {

            jsonObj = JSON.parse(response);

            if (jsonObj.prediction[0]) showFileAnalysis(jsonObj);
            else {

                //console.log(jsonObj.prediction[0]);
                requestStreamAnalysis();

            }


        },
        error: function (error) {

            console.log(error);
            requestStreamAnalysis();
        }
    });

}


/////////////////////////  DO AFTER LOAD ////////////////////////////
$( document ).ready(function() {

    // For now, we need to click the canvas in order to start the visualization
    //$('#spec').click(function() {

    // Adjust canvas size
    $("#spec").width($("#spec-holder").width());
    $("#spec").height($( window ).height() * 0.4);

    // Start audio
    console.log('Starting playback...');
    var base_canvas = document.getElementById('spec');
    var aud = document.getElementById('player');        
    aud.play();        
    
    // Start spectrogram viewer
    var viewer = new AudioViewer(base_canvas, aud, 1024, 1024, $('#spec').width(), $('#spec').height());
    //});

    // Request analysis results every second
    requestStreamAnalysis();

    // Fallback function to restart requests if anything goes wrong
    setInterval('restartRequests()', 2000);

    // Set emergency reload to prevent freezes
    setInterval(function() {

        document.location.reload(true)
        //var aud = document.getElementById('player');        
        //aud.pause();
        //aud.load();
        //aud.play(); 
    
    }, 30 * 60 * 1000); 
    
});