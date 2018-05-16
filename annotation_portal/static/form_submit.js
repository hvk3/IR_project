document.addEventListener('DOMContentLoaded', function(){
	var baseURL = 'http://localhost:5000/'
	document.getElementById("subrank").addEventListener(
		"click",
		function(event){
			var option_a = document.getElementById('a').value;
			var option_b = document.getElementById('b').value;
			var option_c = document.getElementById('c').value;
			var option_d = document.getElementById('d').value;
			var video = document.getElementsByTagName('video')[0].src;
			if (option_a.length > 0 && option_b.length > 0 && option_c.length > 0 && option_d.length > 0)
			{
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
			}
			else
			{
				document.getElementById('ranking').innerHTML = 'Please specify a ranking for every title';
				document.getElementById('ranking').style.display = 'block';
				document.getElementById('ranking').style.color = 'red';
			}
			return false;
		},
		false
	);
});
