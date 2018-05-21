/**
 * Possible parameters for request:
 *  action: "xhttp" for a cross-origin HTTP request
 *  method: Default "GET"
 *  url   : required, but not validated
 *  data  : data to send in a POST request
 *
 * The callback function is called upon completion of the request */
chrome.runtime.onMessage.addListener(function(request, sender, callback) {
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open( "GET", request.url + request.data, false );
    xmlHttp.send( null );
    callback(xmlHttp.responseText);
});
