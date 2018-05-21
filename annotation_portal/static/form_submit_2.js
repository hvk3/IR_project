document.addEventListener('DOMContentLoaded', function(){
	var baseURL = 'http://'+window.location.hostname+':'+window.location.port+'/'
	document.getElementById('video_player').playbackRate = 2;
	document.getElementById("subrank").addEventListener(
		"click",
		function(event){
			var option_a = $("#a").index() + 1;//document.getElementById('a').value;
			var option_b = $("#b").index() + 1;//document.getElementById('b').value;
			var option_c = $("#c").index() + 1;//document.getElementById('c').value;
			var option_d = $("#d").index() + 1;//document.getElementById('d').value;
			// console.log(option_a + "," + option_b + "," + option_c + "," + option_d);
			var video = document.getElementsByTagName('video')[0].src;
			var request = new XMLHttpRequest();
			var serverUrl = baseURL + "update";
			var method = "POST";
			var params = JSON.stringify({"video": video, "a": option_a, "b": option_b, "c": option_c, "d": option_d});

			request.open(method, serverUrl, true);
			request.setRequestHeader("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8");
			request.onreadystatechange = function() {
				if (this.readyState === 4 && this.status === 200) {
					var resp = JSON.parse(this.responseText);
					for (var i in resp)
						idx = resp[i];
					var request_ = new XMLHttpRequest();
					var serverUrl_ = baseURL + '?idx=' + idx;
					var method = "GET";
					request_.open(method, serverUrl_, true);
					request_.setRequestHeader("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8");
					request_.send();
					document.location.href = serverUrl_
				}
			};
			request.send(params)
			return false;
		},
		false
	);
});

