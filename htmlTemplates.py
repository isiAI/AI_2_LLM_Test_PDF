css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://www.helvetia.com/de/web/de/ueber-uns/blog-medien/medien/medienbibliothek/logos/_jcr_content/standardparsys-01/textimage/image.1553513163643.transform-fp/480x270/helvetia-logo-ohne-claim.jpg" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://www.helvetia.com/de/web/de/privatkunden/_jcr_content/homeparsys-01/stage_copy/image.1693900333980.transform-fp/1920x1080/tierkrankenversicherung-telemedizin.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
