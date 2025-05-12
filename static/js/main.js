document.addEventListener('DOMContentLoaded', function() {
    const emotionDisplay = document.getElementById('emotion-display');
    
    // 定期更新情绪显示
    setInterval(() => {
        const video = document.querySelector('img');
        if (video) {
            // 这里可以添加与后端的实时通信逻辑
            // 目前情绪显示由后端直接绘制在视频流上
        }
    }, 1000);
}); 