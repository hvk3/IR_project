//Listen for when a Tab changes state
chrome.tabs.onUpdated.addListener(function(tabId, changeInfo, tab){
    if(changeInfo && changeInfo.status == "complete"){
    	// If video page
    	if(tab.url.includes("watch?v=")){
    		var url = new URL(tab.url);
    		// Extract video ID
			var video_id = url.searchParams.get("v");
    		chrome.tabs.sendMessage(tabId, {data: video_id}, function(response) {
            console.log(response);
        });
    	}
    }
});
