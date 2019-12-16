function AudioViewer(canvas, aud, fftsize, bufsize, hsize) {

    var sourceNode;
    var analyzer;
    var isplaying = 0;
    var speed = 1;
    var colormap;

    // Prepare the canvas
    canvas.width = hsize;
    canvas.height = fftsize / 2;

    // Get the context from the canvas to draw on
    var sg_ctx = canvas.getContext("2d");

    // Create a temp canvas we use for copying
    var tempCanvas = document.createElement("canvas");
    var tempCtx = tempCanvas.getContext("2d");
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;

    // Create color maps used for color distribution
    var colormaps = new Object();
    colormaps.magma = new chroma.ColorScale({
        colors:['#190c3d', '#560f6d', '#be3853', '#fa9107', '#f3e35a', '#f8fb99'],
        positions:[0, .5, .66, .75, .9, 1],
        mode:'rgb',
        limits:[32, 240]
    });

    colormap = colormaps.magma;

    // Create the audio context (chrome only for now)
    if (!window.AudioContext) {
        if (! window.webkitAudioContext) {
            alert('No audiocontext found. Please use the Chrome Browser.');
        }
        window.AudioContext = window.webkitAudioContext;
    }
    var context = new AudioContext();

    // setup a javascript node
    var javascriptNode = context.createScriptProcessor(bufsize, 1, 1);

    // Connect to destination, else it isn't called
    javascriptNode.connect(context.destination);

    javascriptNode.onaudioprocess = function (event)
    {
        var buf = event.inputBuffer;
        var inputData = buf.getChannelData(0);

        // get the average for the first channel
        var array = new Uint8Array(analyzer.frequencyBinCount);
        analyzer.getByteFrequencyData(array);

        // draw the spectrogram
        if (sourceNode.playbackState == sourceNode.PLAYING_STATE) {
            drawSpectrogram(array);
        }
    }

    // setup a analyzer
    analyzer = context.createAnalyser();
    analyzer.smoothingTimeConstant = 0;
    analyzer.fftSize = fftsize;
    analyzer.minDecibels = -150;

    slow_array = new Uint8Array(analyzer.frequencyBinCount);

    // Create a buffer source node
    mediaElement = aud;
    sourceNode = context.createMediaElementSource(mediaElement);

    mediaElement.onplay=function() {isplaying=1;}
    mediaElement.onpause=function() {isplaying=0;}

    mediaElement.oncanplay = function()
    {
        isplaying = !this.paused;
        sourceNode.connect(analyzer);
        analyzer.connect(javascriptNode);
        sourceNode.connect(context.destination);
    }

    var drawSpectrogram = function (array)
    {

        if (!isplaying) {

            // Stop requesting analysis
            //pauseRequests();

            // Cancel drawing
            //return;
        }

        // Resume analysis requests
        //resumeRequests();

        //clear spectrum area
        sg_ctx.clearRect(hsize, 0, 0, canvas.height);

        // copy the current canvas onto the temp canvas
        tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height);

        for (var i = 0; i < array.length; i++) {

            // Draw each pixel with the specific color
            sg_ctx.fillStyle = colormap.getColor(array[i]).hex();

            // Draw the line at the right side of the canvas
            sg_ctx.fillRect(hsize - speed, array.length - 1 - i, speed, 1);

        }
        // Set translate on the canvas
        sg_ctx.translate(-speed, 0);

        // Draw the copied image
        sg_ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height, 0, 0, canvas.width, canvas.height);

        // Reset the transformation matrix
        sg_ctx.setTransform(1, 0, 0, 1, 0, 0);

        // Update detection marker
        //updateDetectionMarker(speed);

    }

}
