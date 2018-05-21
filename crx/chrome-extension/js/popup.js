// By : iamgroot42

function formatParams( params ){
  return "?" + Object
        .keys(params)
        .map(function(key){
          return key+"="+encodeURIComponent(params[key])
        })
        .join("&")
}

// Updata progress bar rating according to score returned by backend
var update_progress_bar = function(videoID, title){
  // Send request to backend server, requesting for score
  params = { 
    "video_id": videoID, 
    "suggested_title": title
  };

  var url = "https://82fcc59b.ngrok.io/get_similarity" + formatParams(params)

  var req = new XMLHttpRequest();
  req.responseType = 'json';
  req.open('GET', url, true);
  req.onload  = function() {
    var jsonResponse = req.response;
    console.log(jsonResponse);
    var score = 0.5;
    var error_div = document.getElementById('not_found');
    if (jsonResponse['score'] != undefined){
      score = jsonResponse['score'];
      error_div.style.visibility = 'hidden'; 
    }
    else{
      // Unhide 'no record' element
      error_div.style.visibility = 'visible';
    }
    // Receive score, set score accordingly
    var ratingBar = document.getElementById('rating_bar');
    rating_bar.setAttribute("aria-valuenow", score * 100);
    rating_bar.style.width = String(score * 100) + "%";
    if (score <= 0.25){
      rating_bar.className = 'progress-bar progress-bar-danger';
    }
    else if (score <= 0.5){
      rating_bar.className = 'progress-bar progress-bar-warning';
    }
    else if (score <= 0.75){
      rating_bar.className = 'progress-bar progress-bar-info';
    }
    else{
      rating_bar.className = 'progress-bar progress-bar-success';
    }    
    console.log("Progress bar updated");
  };
  req.send(null);
};

// Add event listener for button click
document.addEventListener('DOMContentLoaded', function() {
  var logoutButton = document.getElementById('_submit');
  logoutButton.addEventListener('click', function(){
    // Retrieve video ID
    chrome.storage.sync.get('videoID', function (obj) {
        var videoID = obj['videoID'];
        // Get suggested title from input
        var given_title = document.getElementById("suggested_title").value;
        // Update progress bar
        update_progress_bar(videoID, given_title);
        console.log("Updated score bar");
      });
  }, false);
}, false);


chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    // Set current video ID for later use 
    chrome.storage.sync.set({'videoID':request.data},function()
    {
      console.log("Url change detetected. New Video ID: " + request.data);
      // Display default score (that is, score with given title)
      // update_progress_bar(request.data, "");
    });
});
