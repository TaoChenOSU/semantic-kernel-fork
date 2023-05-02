// Copyright (c) Microsoft. All rights reserved.

using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using SemanticKernel.Service.Model;
using SemanticKernel.Service.Storage;

namespace SemanticKernel.Service.Controllers;

/// <summary>
/// Controller for chat invitations.
/// This controller is responsible for:
/// 1. Creating invitation links.
/// 2. Accepting/rejecting invitation links.
/// 3. Managing participants in a chat session.
/// </summary>
[ApiController]
[Authorize]
public class ChatInvitationController : ControllerBase
{
    private readonly ILogger<ChatInvitationController> _logger;
    private readonly ChatInvitationRepository _chatInvitationRepository;
    private readonly ChatSessionRepository _chatSessionRepository;

    /// <summary>
    /// Initializes a new instance of the <see cref="ChatInvitationController"/> class.
    /// </summary>
    /// <param name="logger">The logger.</param>
    /// <param name="chatInvitationRepository">The chat invitation repository.</param>
    /// <param name="chatSessionRepository">The chat session repository.</param>
    public ChatInvitationController(
        ILogger<ChatInvitationController> logger,
        ChatInvitationRepository chatInvitationRepository,
        ChatSessionRepository chatSessionRepository)
    {
        this._logger = logger;
        this._chatInvitationRepository = chatInvitationRepository;
        this._chatSessionRepository = chatSessionRepository;
    }

    /// <summary>
    /// Create a new chat invitation.
    /// </summary>
    /// <returns></returns>
    [HttpPost]
    [Route("chatInvitation/invite")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    public async Task<IActionResult> CreateChatInvitationAsync([FromBody] ChatInvitation chatInvitationParameters)
    {
        var chatId = chatInvitationParameters.ChatId;
        var userId = chatInvitationParameters.UserId;
        // Make sure the chat session exists.
        try
        {
            _ = await this._chatSessionRepository.FindByIdAsync(chatId);
        }
        catch (Exception ex) when (ex is ArgumentOutOfRangeException || ex is KeyNotFoundException)
        {
            this._logger.LogError(ex, "Failed to create chat invitation.");
            return this.NotFound("Chat session does not exist.");
        }

        // Make sure the user hasn't already been invited to the chat session.
        if (await this.HasUserJoinedTheChatSessionAsync(chatId, userId))
        {
            this._logger.LogError("User {0} has joined the chat session {1}.", userId, chatId);
            return this.BadRequest("User has joined the chat session.");
        }

        // Create the invitation.
        var chatInvitation = await this.GetOrCreateChatInvitationAsync(chatId, userId);

        Uri invitationLink = this.GetInvitationLink(chatInvitation, this.GetBaseUri());
        return this.Ok(invitationLink);
    }

    [HttpGet]
    [Route("chatInvitation/accept/{invitationId:Guid}")]
    [ProducesResponseType(StatusCodes.Status302Found)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    public async Task<IActionResult> AcceptChatInvitationAsync(Guid invitationId)
    {
        try
        {
            var chatInvitation = await this._chatInvitationRepository.FindByIdAsync(invitationId.ToString());
            if (chatInvitation.IsAccepted)
            {
                return this.BadRequest("Chat invitation has already been accepted.");
            }

            // TODO: Make sure the user who is accepting the invitation is the same user who was invited.

            // Accept the invitation.
            chatInvitation.IsAccepted = true;
            await this._chatInvitationRepository.UpdateAsync(chatInvitation);

            return this.Ok("You have been added to the chat session. Head to the app to start chatting now!");
        }
        catch (Exception ex) when (ex is ArgumentOutOfRangeException || ex is KeyNotFoundException)
        {
            this._logger.LogError(ex, "Failed to accept chat invitation.");
            return this.NotFound("Chat invitation does not exist.");
        }
    }

    /// <summary>
    /// Returns true if the user has joined the chat session.
    /// </summary>
    /// <param name="chatId"></param>
    /// <param name="userId"></param>
    /// <returns>True if the user has been invited to the cha. False otherwise.</returns>
    private async Task<bool> HasUserJoinedTheChatSessionAsync(string chatId, string userId)
    {
        var invitations = await this._chatInvitationRepository
            .FindInvitationByUserIdAndChatIdAsync(userId, chatId);

        return invitations.Any() && invitations.First().IsAccepted;
    }

    /// <summary>
    /// Get or create a chat invitation. If the chat invitation already exists, it will be returned.
    /// Note that the created chat invitation will be saved to the database.
    /// </summary>
    /// <param name="chatId">The chat id.</param>
    /// <param name="userId">The user id.</param>
    /// <returns>A chat invitation.</returns>
    private async Task<ChatInvitation> GetOrCreateChatInvitationAsync(string chatId, string userId)
    {
        var invitations = await this._chatInvitationRepository
            .FindInvitationByUserIdAndChatIdAsync(userId, chatId);

        if (!invitations.Any())
        {
            var chatInvitation = new ChatInvitation(userId, chatId);
            await this._chatInvitationRepository.CreateAsync(chatInvitation);
            return chatInvitation;
        }

        return invitations.First();
    }

    /// <summary>
    /// Get the base uri for the request.
    /// </summary>
    /// <returns>The base uri.</returns>
    private Uri GetBaseUri()
    {
        var request = this.Request;
        var baseUri = new Uri($"{request.Scheme}://{request.Host}{request.PathBase}");
        return baseUri;
    }

    /// <summary>
    /// Get the invitation link for a chat invitation.
    /// </summary>
    /// <param name="chatInvitation">The invitation.</param>
    /// <param name="requestUri">The base uri.</param>
    /// <returns></returns>
    private Uri GetInvitationLink(ChatInvitation chatInvitation, Uri requestUri)
    {
        var invitationLink = new Uri(requestUri, $"chatInvitation/accept/{chatInvitation.Id}");
        return invitationLink;
    }
}