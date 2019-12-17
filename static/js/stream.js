/////////////////////////  DO AFTER LOAD ////////////////////////////
$( document ).ready(function() {

    // For now, we need to click the canvas in order to start the visualization
    //$('#spec').click(function() {
        console.log('Starting playback...');
        var base_canvas = document.getElementById('spec');
        var aud = document.getElementById('player');        
        aud.play();

        // Adjust canvas size
        $("#spec").width($("#spec-holder").width());
        $("#spec").height($( window ).height() * 0.4);
        
        // Start spectrogram viewer
        var viewer = new AudioViewer(base_canvas, aud, 1024, 1024, $('#spec').width());
    //});

    
});